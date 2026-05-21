import time
import os
from typing import Optional
from safetensors.torch import load_file, save_file

import torch
from tqdm import tqdm


def _base_key(n: str) -> str:
    return n.replace("lora_A.", "").replace("lora_B.", "")


def _return_deltas_by_key(tv):
    deltas_by_key: dict[str, list[torch.Tensor]] = {}
    for _, params in tv.items():
        # Collect keys
        # keys_delta: those not related to LoRA but full weights (if any)
        keys_delta = {k for k in params if not ("lora_A." in k or "lora_B." in k)}
        keys_A = {k for k in params if "lora_A." in k}
        keys_B = {k for k in params if "lora_B." in k}
        common = {_base_key(k) for k in keys_A} & {_base_key(k) for k in keys_B}
        for bk in common:
            a_key = next(k for k in keys_A if _base_key(k) == bk)
            b_key = next(k for k in keys_B if _base_key(k) == bk)
            original_device = params[b_key].device
            params_B = params[b_key].to("cuda")
            params_A = params[a_key].to("cuda")
            delta = (params_B @ params_A).to(original_device)
            deltas_by_key.setdefault(bk, []).append(delta)
        for dk in keys_delta:
            delta = params[dk]
            deltas_by_key.setdefault(dk, []).append(delta)

    return deltas_by_key


# ------------------------------
# Linear
# ------------------------------


def Linear(
    tv: dict[str, dict[str, torch.Tensor]],
    merging_type: str = "sum",  # 'sum' | 'average'
    scaling_factor: float = 1.0,
    cache_path: Optional[str] = None,
    profile: bool = False,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and os.path.exists(cache_path) and not profile:
        print(f"Loading cached merged TV from {cache_path}...")
        # Load safetensor
        merged = load_file(cache_path)
        for k in merged.keys():
            merged[k] = merged[k] * scaling_factor
        return merged
    start_time = time.time()
    A_acc: dict[str, torch.Tensor] = {}
    B_acc: dict[str, torch.Tensor] = {}
    for _, params in tv.items():
        for name, t in params.items():
            if "lora_A." in name:
                k = _base_key(name)
                A_acc[k] = t.clone() if k not in A_acc else A_acc[k].add_(t)
            elif "lora_B." in name:
                k = _base_key(name)
                B_acc[k] = t.clone() if k not in B_acc else B_acc[k].add_(t)

    merged: dict[str, torch.Tensor] = {}
    for k in A_acc.keys() & B_acc.keys():
        merged[k] = B_acc[k] @ A_acc[k]
    if merging_type == "mean":
        n = len(tv)
        if n > 0:
            for k in merged:
                merged[k].div_(n)

    end_time = time.time()
    # print some info about the merging process
    print(
        f"Linear merging with merging_type={merging_type}, "
        f"scaling_factor={scaling_factor} took {end_time - start_time:.2f} seconds."
    )
    if cache_path is not None and not profile:
        print(f"Caching merged TV to {cache_path}...")
        # Save safetensor
        save_file(merged, cache_path)
    # apply scaling factor
    for k in A_acc.keys() & B_acc.keys():
        merged[k] = merged[k] * scaling_factor
    return merged


# ------------------------------
# Task Arithmetic
# ------------------------------


def TA(
    tv: dict[str, dict[str, torch.Tensor]],
    merging_type: str = "sum",  # 'sum' | 'mean'
    scaling_factor: float = 0.3,
    cache_path: Optional[str] = None,
    profile: bool = False,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and os.path.exists(cache_path) and not profile:
        print(f"Loading cached merged TV from {cache_path}...")
        # Load safetensor
        merged = load_file(cache_path)
        for k in merged.keys():
            merged[k] = merged[k] * scaling_factor
        return merged
    start_time = time.time()
    merged: dict[str, torch.Tensor] = {}
    for _, params in tv.items():
        keys_A = {k for k in params if "lora_A." in k}
        keys_B = {k for k in params if "lora_B." in k}
        common = {_base_key(k) for k in keys_A} & {_base_key(k) for k in keys_B}
        for bk in common:
            a_key = next(k for k in keys_A if _base_key(k) == bk)
            b_key = next(k for k in keys_B if _base_key(k) == bk)
            delta = params[b_key] @ params[a_key]
            if bk not in merged:
                merged[bk] = delta.clone()
            else:
                merged[bk].add_(delta)
    if merging_type == "mean":
        n = len(tv)
        if n > 0:
            for k in merged:
                merged[k].div_(n)
    end_time = time.time()
    # print some info about the merging process
    print(
        f"TA merging with merging_type={merging_type}, "
        f"scaling_factor={scaling_factor} took {end_time - start_time:.2f} seconds."
    )
    if cache_path is not None and not profile:
        print(f"Caching merged TV to {cache_path}...")
        # Save safetensor
        save_file(merged, cache_path)
    # apply scaling factor
    for k in merged.keys():
        merged[k] = merged[k] * scaling_factor
    return merged


# ------------------------------
# SVD (shared U, s; plain mean/sum over sV)
# ------------------------------


def SVD(
    tv: dict[str, dict[str, torch.Tensor]],
    merging_type: str = "sum",  # 'mean' | 'sum'
    svd_tol: float = 1e-5,
    scaling_factor: float = 1.0,
    cache_path: Optional[str] = None,
    profile: bool = False,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and os.path.exists(cache_path) and not profile:
        print(f"Loading cached merged TV from {cache_path}...")
        # Load safetensor
        merged = load_file(cache_path)
        for k in merged.keys():
            merged[k] = merged[k] * scaling_factor
        return merged
    start_time = time.time()
    deltas_by_key = _return_deltas_by_key(tv)

    merged: dict[str, torch.Tensor] = {}
    T = len(tv)

    for bk, lst in deltas_by_key.items():
        R, C = lst[0].shape
        M = torch.cat(lst, dim=1)
        U, s, Vh = torch.linalg.svd(M.to(torch.float64), full_matrices=False)
        U, s, Vh = U.to(torch.float32), s.to(torch.float32), Vh.to(torch.float32)

        keep = s > svd_tol
        if keep.sum() == 0:
            merged[bk] = torch.zeros_like(lst[0])
            continue
        U = U[:, keep]
        s = s[keep]
        Vh = Vh[keep, :]

        s_diag = torch.diag(s)
        sV_tasks = []
        for t in range(T):
            Vh_t = Vh[:, t * C : (t + 1) * C]
            sV_t = s_diag @ Vh_t
            sV_tasks.append(sV_t)
        S = torch.stack(sV_tasks, dim=0)

        if merging_type == "sum":
            sV_merged = S.sum(dim=0)
        else:
            sV_merged = S.mean(dim=0)

        delta_merged = U @ sV_merged
        merged[bk] = delta_merged
    end_time = time.time()
    # print some info about the merging process
    print(
        f"SVD merging with svd_tol={svd_tol}, "
        f"merging_type={merging_type}, scaling_factor={scaling_factor} "
        f"took {end_time - start_time:.2f} seconds."
    )
    if cache_path is not None and not profile:
        print(f"Caching merged TV to {cache_path}...")
        # Save safetensor
        save_file(merged, cache_path)
    for k in merged.keys():
        merged[k] = merged[k] * scaling_factor

    return merged


# ------------------------------
# Ties + DareTies
# ------------------------------


def Ties(
    tv: dict[str, dict[str, torch.Tensor]],
    merging_type: str = "sum",
    topK: float = 1.0,
    drop_p: float = 0.0,
    seed: Optional[int] = None,
    scaling_factor: float = 0.3,
    cache_path: Optional[str] = None,
    profile: bool = False,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and os.path.exists(cache_path) and not profile:
        print(f"Loading cached merged TV from {cache_path}...")
        # Load safetensor
        merged = load_file(cache_path)
        for k in merged.keys():
            merged[k] = merged[k] * scaling_factor
        return merged

    def _rowwise_topk_mask(x: torch.Tensor, k: int) -> torch.Tensor:
        if k >= x.shape[1]:
            return torch.ones_like(x, dtype=torch.bool)
        _, idx = x.abs().topk(k, dim=1)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return mask

    def _parse_topk(D: int, tk: float) -> int:
        if tk > 1:
            tk = tk / 100.0
        tk = float(max(0.0, min(1.0, tk)))
        return D if tk >= 1.0 else max(1, int(round(D * tk)))

    start_time = time.time()
    deltas_by_key = _return_deltas_by_key(tv)

    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
    keep_p = 1.0 - drop_p
    scale = 1.0 / keep_p if drop_p > 0 else 1.0

    merged: dict[str, torch.Tensor] = {}
    T = len(tv)

    for bk, lst in tqdm(deltas_by_key.items()):
        stack = torch.stack(lst, dim=0)
        flat = stack.reshape(T, -1)

        # Top-K
        k = _parse_topk(flat.shape[1], topK)
        topk_mask = _rowwise_topk_mask(flat, k)
        flat = flat * topk_mask.float()

        # DARE
        if drop_p > 0:
            bern = torch.bernoulli(torch.full_like(flat, keep_p), generator=gen)
            flat = flat * bern * scale

        sign_sum = flat.sign().sum(dim=0)
        sign_sum[sign_sum == 0] = 1
        rows_keep = torch.where(sign_sum.unsqueeze(0) > 0, flat > 0, flat < 0)
        selected = flat * rows_keep.float()

        if merging_type == "mean":
            nonzero = (selected != 0).sum(dim=0).clamp_min(1).float()
            agg = selected.sum(dim=0) / nonzero
        else:  # 'sum'
            agg = selected.sum(dim=0)

        merged[bk] = agg.reshape(stack.shape[1:])
    end_time = time.time()
    # print some info about the merging process
    if drop_p > 0:
        print(
            f"Dare-Ties merging with topK={topK}, drop_p={drop_p}, "
            f"merging_type={merging_type}, seed={seed}, scaling_factor={scaling_factor} "
            f"took {end_time - start_time:.2f} seconds."
        )
    else:
        print(
            f"Ties merging with topK={topK}, "
            f"merging_type={merging_type}, seed={seed}, scaling_factor={scaling_factor} "
            f"took {end_time - start_time:.2f} seconds."
        )
    if cache_path is not None and not profile:
        print(f"Caching merged TV to {cache_path}...")
        # Save safetensor
        save_file(merged, cache_path)

    for k in merged.keys():
        merged[k] = merged[k] * scaling_factor
    return merged


# ------------------------------
# KnOTS (Ties+DareTies)
# ------------------------------


def KnOTS(
    tv: dict[str, dict[str, torch.Tensor]],
    merging_type: str = "sum",  # 'sum' | 'mean'
    drop_p: float = 0.0,
    seed: Optional[int] = None,
    svd_tol: float = 1e-5,
    topK: float = 1.0,
    scaling_factor: float = 0.3,
    cache_path: Optional[str] = None,
    profile: bool = False,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and os.path.exists(cache_path) and not profile:
        print(f"Loading cached merged TV from {cache_path}...")
        # Load safetensor
        merged = load_file(cache_path)
        for k in merged.keys():
            merged[k] = merged[k] * scaling_factor
        return merged
    start_time = time.time()
    deltas_by_key = _return_deltas_by_key(tv)

    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
    keep_p = 1.0 - drop_p
    scale = 1.0 / keep_p if keep_p > 0 else 0.0

    def _rowwise_topk_mask(x: torch.Tensor, k: int) -> torch.Tensor:
        if k >= x.shape[1]:
            return torch.ones_like(x, dtype=torch.bool)
        _, idx = x.abs().topk(k, dim=1)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return mask

    def _parse_topk(TD: int, tk: float) -> int:
        if tk > 1:
            tk = tk / 100.0
        tk = float(max(0.0, min(1.0, tk)))
        return max(1, int(round(TD * tk))) if tk < 1.0 else TD

    merged: dict[str, torch.Tensor] = {}
    T = len(tv)

    for bk, lst in tqdm(deltas_by_key.items()):
        original_device = lst[0].device
        lst = [t.to("cuda") for t in lst]
        R, C = lst[0].shape
        M = torch.cat(lst, dim=1)
        U, s, Vh = torch.linalg.svd(M, full_matrices=False)
        U, s, Vh = U.float(), s.float(), Vh.float()

        keep = s > svd_tol
        if keep.sum() == 0:
            merged[bk] = torch.zeros_like(lst[0])
            continue
        U, s, Vh = U[:, keep], s[keep], Vh[keep, :]

        s_diag = torch.diag(s)
        sV_tasks: list[torch.Tensor] = []
        for t in range(T):
            Vh_t = Vh[:, t * C : (t + 1) * C]
            sV_t = s_diag @ Vh_t
            sV_tasks.append(sV_t)
        S = torch.stack(sV_tasks, dim=0)

        # --- Top-K ---
        flat = S.reshape(T, -1)
        k = _parse_topk(flat.shape[1], topK)
        topk_mask = _rowwise_topk_mask(flat, k)
        flat = flat * topk_mask.float()

        # --- DARE  ---
        if drop_p > 0:
            bern = torch.bernoulli(torch.full_like(flat, keep_p), generator=gen)
            flat = flat * bern * scale

        sign_sum = flat.sign().sum(dim=0)
        sign_sum[sign_sum == 0] = 1
        rows_keep = torch.where(sign_sum.unsqueeze(0) > 0, flat > 0, flat < 0)
        selected = flat * rows_keep.float()

        if merging_type == "mean":
            nonzero = (selected != 0).sum(dim=0).clamp_min(1).float()
            agg = selected.sum(dim=0) / nonzero
        else:  # 'sum'
            agg = selected.sum(dim=0)

        sV_merged = agg.reshape(S.shape[1], S.shape[2])
        merged[bk] = (U @ sV_merged).to(original_device)
    end_time = time.time()

    # print some info about the merging process
    if drop_p > 0:
        print(
            f"KnOTS-Dare-Ties merging with topK={topK}, drop_p={drop_p}, svd_tol={svd_tol}, "
            f"merging_type={merging_type}, seed={seed}, scaling_factor={scaling_factor} "
            f"took {end_time - start_time:.2f} seconds."
        )
    else:
        print(
            f"KnOTS-Ties merging with topK={topK}, svd_tol={svd_tol}, "
            f"merging_type={merging_type}, seed={seed}, scaling_factor={scaling_factor} "
            f"took {end_time - start_time:.2f} seconds."
        )
    if cache_path is not None and not profile:
        print(f"Caching merged TV to {cache_path}...")
        # Save safetensor
        save_file(merged, cache_path)
    for k in merged.keys():
        merged[k] = merged[k] * scaling_factor
    return merged


# ------------------------------
# LoRA-LEGO
# ------------------------------
def LoRA_LEGO(
    tv: dict[str, dict[str, torch.Tensor]],
    K: Optional[int] = None,
    kmeans_iters: int = 10,
    use_param_reweighting: bool = True,
    use_output_reweighting: bool = True,
    seed: Optional[int] = None,
    cache_path: Optional[str] = None,
    profile: bool = False,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and os.path.exists(cache_path) and not profile:
        print(f"Loading cached merged TV from {cache_path}...")
        # Load safetensor
        return load_file(cache_path)
    _EPS = 1e-8

    def _base_AB(params: dict[str, torch.Tensor], bk: str):
        a_key = next(k for k in params if "lora_A." in k and _base_key(k) == bk)
        b_key = next(k for k in params if "lora_B." in k and _base_key(k) == bk)
        return params[a_key], params[b_key]

    def _normalize(v):
        return v / (v.norm() + _EPS)

    def _kmeans(x: torch.Tensor, K: int, iters: int, gen: Optional[torch.Generator]):
        N, D = x.shape
        if K >= N:
            return torch.arange(N, device=x.device), x.clone()
        idx = torch.randperm(N, generator=gen, device=x.device)[:K]
        cent = x[idx].clone()
        for _ in range(iters):
            d = (x.unsqueeze(1) - cent.unsqueeze(0)).pow(2).sum(2)
            assign = d.argmin(1)
            for k in range(K):
                m = assign == k
                if m.any():
                    cent[k] = x[m].mean(0)
        return assign, cent

    start_time = time.time()
    gen = torch.Generator(device="cpu").manual_seed(int(seed)) if seed else None
    merged = {}

    task_keys = list(tv.keys())
    bases_A = [{_base_key(k) for k in tv[t] if "lora_A." in k} for t in task_keys]
    bases_B = [{_base_key(k) for k in tv[t] if "lora_B." in k} for t in task_keys]
    common_bases = set.intersection(*bases_A) & set.intersection(*bases_B)

    for bk in tqdm(common_bases):
        AB = [_base_AB(tv[t], bk) for t in task_keys]
        r, d_in = AB[0][0].shape
        d_out, rB = AB[0][1].shape
        assert r == rB
        KK = K or r

        # build raw pool
        feat_list, raw_list = [], []
        for A, B in AB:
            for k in range(r):
                a_vec = A[k]
                b_vec = B[:, k]
                feat = torch.cat([_normalize(a_vec), _normalize(b_vec)])
                raw = torch.cat([a_vec, b_vec])
                feat_list.append(feat)
                raw_list.append(raw)

        features = torch.stack(feat_list)
        raw_pool = torch.stack(raw_list)
        labels, _ = _kmeans(features, KK, kmeans_iters, gen)

        merged_delta = torch.zeros((d_out, d_in), dtype=raw_pool.dtype, device=raw_pool.device)
        for cid in range(KK):
            m = labels == cid
            if not m.any():
                continue
            mu = raw_pool[m].mean(0)
            if use_param_reweighting:
                eps = 1e-12
                inf_norm = mu.abs().amax()
                avg_inf = raw_pool[m].abs().amax(dim=1).mean()
                mu = mu * (avg_inf / (inf_norm + eps))
            a_bar = mu[:d_in]
            b_bar = mu[d_in:]
            merged_delta.add_(torch.outer(b_bar, a_bar))

        if use_output_reweighting and KK > 0:
            merged_delta = merged_delta * ((r**0.5) / (KK**0.5))

        merged[bk] = merged_delta
    end_time = time.time()

    # print some info about the merging process
    print(
        f"LoRA-LEGO merging with K={K}, kmeans_iters={kmeans_iters}, "
        f"use_param_reweighting={use_param_reweighting}, use_output_reweighting={use_output_reweighting}, seed={seed}"  # noqa
        f" took {end_time - start_time:.2f} seconds."
    )

    if cache_path is not None and not profile:
        print(f"Caching merged TV to {cache_path}...")
        # Save safetensor
        save_file(merged, cache_path)

    return merged


def ISO_C(
    tv: dict[str, dict[str, torch.Tensor]],
    scaling_factor: float,
    cache_path: Optional[str] = None,
    profile: bool = False,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and os.path.exists(cache_path) and not profile:
        print(f"Loading cached merged TV from {cache_path}...")
        # Load safetensor
        merged = load_file(cache_path)
        for k in merged.keys():
            merged[k] = merged[k] * scaling_factor
        return merged
    start_time = time.time()
    deltas = _return_deltas_by_key(tv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_device = (list(deltas.values()))[0][0].device
    print("Computing SVD...")
    with torch.no_grad():
        merged = {}
        for key, tensor_list in tqdm(deltas.items(), ncols=80):
            tvs = [tv.to(device) for tv in tensor_list]
            merged[key] = sum(tvs) / len(tvs)

            if len(tvs[0].shape) == 2 and "text_projection" not in key:
                merged[key] *= len(tvs)
                U, S, V = torch.linalg.svd(merged[key], full_matrices=False)
                S_mean = torch.ones_like(S) * S.mean()

                merged[key] = torch.linalg.multi_dot(
                    (
                        U,
                        torch.diag(S_mean),
                        V,
                    )
                ).to(original_device)
    end_time = time.time()

    # print some info about the merging process
    print(
        f"ISO-C merging with scaling_factor={scaling_factor},"
        f" took {end_time - start_time:.2f} seconds."
    )
    if cache_path is not None and not profile:
        print(f"Caching merged TV to {cache_path}...")
        # Save safetensor
        save_file(merged, cache_path)

    for k in merged.keys():
        merged[k] = merged[k] * scaling_factor

    return merged


@torch.no_grad()
def ISO_CTS(
    tv: dict[str, dict[str, torch.Tensor]],
    scaling_factor: float = 1.0,
    datasets: Optional[list[str]] = None,
    common_space_fraction: float = 0.8,
    cache_path: Optional[str] = None,
    profile: bool = False,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and os.path.exists(cache_path) and not profile:
        print(f"Loading cached merged TV from {cache_path}...")
        # Load safetensor
        merged = load_file(cache_path)
        for k in merged.keys():
            merged[k] = merged[k] * scaling_factor
        return merged
    start_time = time.time()
    len_datasets = len(tv.keys()) if datasets is None else len(datasets)
    deltas = _return_deltas_by_key(tv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_device = (list(deltas.values()))[0][0].device
    merged = {}

    print("Computing SVD...")
    for key in tqdm(deltas.keys()):
        shape_ = deltas[key][0].shape

        is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in key)
        if not is_2d_matrix:
            # print(f"Combining by avg {key}...")
            for i, task_specific_tv in enumerate(deltas[key]):
                vec = task_specific_tv.to(device)
                if i == 0:
                    merged[key] = vec.clone()
                else:
                    merged[key] += (vec - merged[key]) / (i + 1)
            continue

        # print(f"Computing common space using sum for {key}...")
        combined_w = sum([task_vector.to(device) for task_vector in deltas[key]])

        ### Calculate the common space size (making sure that task specific space is equally divisible) ###
        common_space_index_s = int(min(shape_) * common_space_fraction)
        _task_specific_total_space_index_s = (
            round((min(shape_) - common_space_index_s) / len_datasets) * len_datasets
        )
        common_space_index_s = min(shape_) - _task_specific_total_space_index_s

        u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
        common_space_u = u[:, :common_space_index_s]
        common_space_s = s[:common_space_index_s]
        common_space_v = v[:common_space_index_s, :]
        ###################################################################

        ### Calculate task specific space ###
        n_dims_per_task = int((min(shape_) - common_space_index_s) / len_datasets)
        for i, task_vector in enumerate(deltas[key]):
            w = task_vector.to(device)

            # calculate the projection onto task specific space to remove the common space
            w_ts = w - common_space_u @ common_space_u.T @ w
            u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)

            if i == 0:
                combined_space_u = torch.zeros_like(u_ts, device=device)
                combined_space_s = torch.zeros_like(s_ts, device=device)
                combined_space_v = torch.zeros_like(v_ts, device=device)

            combined_space_u[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = u_ts[
                :, :n_dims_per_task
            ]
            combined_space_s[i * n_dims_per_task : (i + 1) * n_dims_per_task] = s_ts[
                :n_dims_per_task
            ]
            combined_space_v[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = v_ts[
                :n_dims_per_task, :
            ]
        ###################################################################

        combined_space_u[
            :,
            len_datasets * n_dims_per_task : len_datasets * n_dims_per_task + common_space_index_s,
        ] = common_space_u
        combined_space_s[
            len_datasets * n_dims_per_task : len_datasets * n_dims_per_task + common_space_index_s
        ] = common_space_s
        combined_space_v[
            len_datasets * n_dims_per_task : len_datasets * n_dims_per_task + common_space_index_s,
            :,
        ] = common_space_v

        ### Orthogonalize combined_space_u and combined_space_v ###
        u_combined_space_u, s_combined_space_u, v_combined_space_u = torch.linalg.svd(
            combined_space_u, full_matrices=False
        )
        u_combined_space_v, s_combined_space_v, v_combined_space_v = torch.linalg.svd(
            combined_space_v, full_matrices=False
        )
        combined_space_u = u_combined_space_u @ v_combined_space_u
        combined_space_v = u_combined_space_v @ v_combined_space_v
        ###################################################################

        combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()

        merged[key] = torch.linalg.multi_dot(
            (
                combined_space_u,
                torch.diag(combined_space_s),
                combined_space_v,
            )
        )
        merged[key] = merged[key].to(original_device)
    end_time = time.time()
    # print some info about the merging process
    print(
        f"ISO-CTS merging with scaling_factor={scaling_factor},"
        f" took {end_time - start_time:.2f} seconds."
    )
    if cache_path is not None and not profile:
        print(f"Caching merged TV to {cache_path}...")
        # Save safetensor
        save_file(merged, cache_path)
    for k in merged.keys():
        merged[k] = merged[k] * scaling_factor
    return merged


def TSVM(
    tv: dict[str, dict[str, torch.Tensor]],
    datasets: Optional[list[str]] = None,
    scaling_factor: float = 1.0,
    cache_path: Optional[str] = None,
    profile: bool = False,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and os.path.exists(cache_path) and not profile:
        print(f"Loading cached merged TV from {cache_path}...")
        # Load safetensor
        merged = load_file(cache_path)
        for k in merged.keys():
            merged[k] = merged[k] * scaling_factor
        return merged
    start_time = time.time()
    len_datasets = len(tv.keys()) if datasets is None else len(datasets)
    sv_reduction = 1 / len_datasets
    deltas = _return_deltas_by_key(tv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_device = (list(deltas.values()))[0][0].device
    print("Computing SVD...")
    with torch.no_grad():
        merged = {}
        for key in tqdm(deltas.keys(), ncols=80):
            merged[key] = {}
            for i, task_vector in enumerate(deltas[key]):
                vec = task_vector.to(device)

                if len(task_vector.shape) == 2 and "text_projection" not in key:
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[:reduced_index_s]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                else:
                    if i == 0:
                        merged[key] = vec.clone()
                    else:
                        merged[key] += (vec - merged[key]) / (i + 1)

            if len(task_vector.shape) == 2 and "text_projection" not in key:
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

                merged[key] = (
                    torch.linalg.multi_dot(
                        (
                            u_u,
                            v_u,
                            torch.diag(sum_s),
                            u_v,
                            v_v,
                        )
                    )
                ).to(original_device)

    end_time = time.time()
    # print some info about the merging process
    print(
        f"TSVM merging with scaling_factor={scaling_factor},"
        f" took {end_time - start_time:.2f} seconds."
    )
    if cache_path is not None and not profile:
        print(f"Caching merged TV to {cache_path}...")
        # Save safetensor
        save_file(merged, cache_path)
    for k in merged.keys():
        merged[k] = merged[k] * scaling_factor
    return merged


def RobustMerge(
    tv: dict[str, dict[str, torch.Tensor]],
    mask_ratio: float = 0.8,  # This is 'k' in the algorithm (pruning rate)
    fuse_weight: float = 2.0,  # This is lambda
    scaling_factor: float = 1.0,  # Standard LoRA scaling (alpha/rank)
    cache_path: Optional[str] = None,
    profile: bool = False,
) -> dict[str, torch.Tensor]:
    # ... (Cache loading logic remains the same) ...
    if cache_path is not None and os.path.exists(cache_path) and not profile:
        print(f"Loading cached merged TV from {cache_path}...")
        # Placeholder for load logic
        merged = load_file(cache_path)
        for k in merged.keys():
            merged[k] = merged[k] * scaling_factor
        return merged

    start_time = time.time()

    # 1. Identify unique base keys (e.g., "attn.q") to pair lora_A and lora_B
    all_keys = set().union(*[m.keys() for m in tv.values()])

    # Helper to group keys.
    # We want to find pairs like "layer.weight.lora_A" and "layer.weight.lora_B"
    # and map them to "layer.weight"
    lora_groups = {}
    other_keys = []

    for key in all_keys:
        if "lora_A" in key:
            base_key = key.replace("lora_A.", "").replace("lora_B.", "")  # simple strip
            # A more robust regex might be needed depending on naming convention
            # usually: name.lora_A.weight -> base: name.weight
            # if key.endswith(".weight"):
            #     base_key = key.replace(".lora_A.weight", "")

            if base_key not in lora_groups:
                lora_groups[base_key] = {}
            lora_groups[base_key]["A"] = key

        elif "lora_B" in key:
            base_key = key.replace("lora_B.", "").replace("lora_A.", "")
            # if key.endswith(".weight"):
            #     base_key = key.replace(".lora_B.weight", "")

            if base_key not in lora_groups:
                lora_groups[base_key] = {}
            lora_groups[base_key]["B"] = key
        else:
            other_keys.append(key)

    final_merged: dict[str, torch.Tensor] = {}

    # --- Process Non-LoRA keys (Average them) ---
    print("Merging non-LoRA keys...")
    for key in other_keys:
        tensors = [model[key] for model in tv.values() if key in model]
        if tensors:
            final_merged[key] = torch.stack(tensors).mean(dim=0)

    # --- Process LoRA Groups (Algorithm 1) ---
    print("Merging LoRA modules using Algorithm 1...")

    # We iterate over the BASE layers (e.g., 'attn.q')
    for base_key, pair in tqdm(lora_groups.items()):
        key_A = pair.get("A")
        key_B = pair.get("B")

        if not key_A or not key_B:
            continue  # Skip if incomplete pair

        # Stack tensors from all models: Shape [N, r, in] and [N, out, r]
        # Filter out models that might miss this specific key
        models_with_key = [m for m in tv.values() if key_A in m and key_B in m]
        if not models_with_key:
            continue

        A_list = [m[key_A] for m in models_with_key]
        B_list = [m[key_B] for m in models_with_key]

        A_stack = torch.stack(A_list, dim=0)  # [N, r, d_in]
        B_stack = torch.stack(B_list, dim=0)  # [N, d_out, r]

        N, r, d_in = A_stack.shape
        _, d_out, _ = B_stack.shape

        # === Step 1: Pruning and Complementary Parameter Scaling ===

        # 1. Pruning A
        # Algorithm: Keep top-k magnitude.
        # Implementation: We zero out the small values.
        flat_A = A_stack.reshape(N, -1)
        k_A = int(flat_A.shape[1] * (1 - mask_ratio))
        kth_val_A, _ = flat_A.abs().kthvalue(k_A, dim=1, keepdim=True)
        mask_A = (flat_A.abs() >= kth_val_A).reshape(N, r, d_in)
        A_tilde = A_stack * mask_A  # \tilde{A}

        # 2. Pruning B
        flat_B = B_stack.reshape(N, -1)
        k_B = int(flat_B.shape[1] * (1 - mask_ratio))
        kth_val_B, _ = flat_B.abs().kthvalue(k_B, dim=1, keepdim=True)
        mask_B = (flat_B.abs() >= kth_val_B).reshape(N, d_out, r)
        B_tilde = B_stack * mask_B  # \tilde{B}

        # 3. Calculate Scalar S based on A (Row-wise for A)
        # A is [N, r, d_in]. We sum over d_in (dim=2)
        sum_abs_A = torch.sum(A_stack.abs(), dim=2)  # [N, r]
        sum_abs_A_tilde = torch.sum(A_tilde.abs(), dim=2)  # [N, r]

        # Avoid div by zero
        s_vector = sum_abs_A / (sum_abs_A_tilde + 1e-6)  # [N, r] (This is S^i in algo)

        # === Step 2: Cross-Task Normalization ===

        # Normalize across models (dim=0)
        # S_n^i = s_n^i / Sum_{n=1 to N} s_n^i
        sum_s_across_models = torch.sum(s_vector, dim=0, keepdim=True)  # [1, r]
        s_tilde = s_vector / (sum_s_across_models + 1e-6)  # [N, r]

        # === Obtain parameter-efficient modules & Merge ===

        # delta_W_n = B_tilde . S_tilde . A_tilde
        # S_tilde is [N, r]. We need to scale the inner dimension 'r'.
        # Visually: B [N, out, r] @ diag(S) @ A [N, r, in]
        # Broadcasting: Multiply A_tilde by s_tilde. s_tilde needs to be [N, r, 1]

        s_tilde_unsqueezed = s_tilde.unsqueeze(-1)  # [N, r, 1]
        scaled_A = A_tilde * s_tilde_unsqueezed

        # Matrix multiplication per model
        delta_W_n = torch.bmm(B_tilde, scaled_A)  # [N, out, in]

        # Sum across all models
        delta_W_sum = torch.sum(delta_W_n, dim=0)  # [out, in]

        # Apply global Lambda (fuse_weight) and LoRA scaling factor
        # Note: Standard LoRA usually requires merged weights to be multiplied by scaling_factor
        final_delta = fuse_weight * delta_W_sum

        # Store result.
        # Note: This returns the DENSE update for the weight matrix.
        # If the original model keys were "layer.weight", this delta should be added to it.
        # Since we are returning a state dict, we return the dense delta.
        final_merged[base_key] = final_delta

    end_time = time.time()
    print(
        f"RobustMerge merging with mask_ratio={mask_ratio}, "
        f"fuse_weight={fuse_weight}, scaling_factor={scaling_factor} took {end_time - start_time:.2f} seconds."
    )
    if cache_path is not None and not profile:
        print(f"Caching merged TV to {cache_path}...")
        # Save safetensor
        save_file(final_merged, cache_path)
    for k in final_merged.keys():
        final_merged[k] = final_merged[k] * scaling_factor

    return final_merged


def EMR(
    tv: dict[str, dict[str, torch.Tensor]],
    scaling_factor: float = 1.0,
    cache_path: Optional[str] = None,
    profile: bool = False,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and os.path.exists(cache_path) and not profile:
        print(f"Loading cached merged TV from {cache_path}...")
        # Load safetensor
        merged = load_file(cache_path)
        for k in merged.keys():
            merged[k] = merged[k] * scaling_factor
        return merged
    start_time = time.time()
    deltas_by_key = _return_deltas_by_key(tv)
    merged: dict[str, torch.Tensor] = {}

    for bk, lst in tqdm(deltas_by_key.items()):
        S = torch.stack(lst, dim=0)
        mean = S.mean(dim=0)

        flag = torch.sign(mean)
        flag[flag == 0] = 1
        flag = flag.to(S.dtype)

        aligned = (S * flag) > 0
        masked_abs = torch.where(aligned, S.abs(), torch.zeros_like(S))
        max_abs, _ = masked_abs.max(dim=0)

        merged[bk] = max_abs * flag

    end_time = time.time()
    # print some info about the merging process
    print(
        f"EMR merging with scaling_factor={scaling_factor},"
        f" took {end_time - start_time:.2f} seconds."
    )
    if cache_path is not None and not profile:
        print(f"Caching merged TV to {cache_path}...")
        # Save safetensor
        save_file(merged, cache_path)
    for k in merged.keys():
        merged[k] = merged[k] * scaling_factor

    return merged


# ------------------------------
# FR-Merging, FREE-Merging
# ------------------------------


def FreeMerge(
    tv: dict[str, dict[str, torch.Tensor]],
    merging_type: str = "sum",
    low_ratio: float = 0.10,
    high_ratio: float = 0.70,
    mode: str = "band+high",  # 'band+high' | 'band' | 'high' | 'low' | 'all'
    scaling_factor: float = 1.0,
    cache_path: Optional[str] = None,
    profile: bool = False,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and os.path.exists(cache_path) and not profile:
        print(f"Loading cached merged TV from {cache_path}...")
        # Load safetensor
        merged = load_file(cache_path)
        for k in merged.keys():
            merged[k] = merged[k] * scaling_factor
        return merged
    start_time = time.time()
    deltas_by_key = _return_deltas_by_key(tv)
    merged: dict[str, torch.Tensor] = {}
    T = len(tv)

    def _fft_filter_2d(weight_2d: torch.Tensor) -> torch.Tensor:
        device = weight_2d.device
        dtype = weight_2d.dtype
        x = weight_2d.to(torch.float32)

        F = torch.fft.fft2(x)
        H, W = F.shape[-2], F.shape[-1]
        mind = min(H, W)

        low_r = max(1, int(mind * float(low_ratio)))
        high_r = max(low_r + 1, int(mind * float(high_ratio)))

        cy, cx = H // 2, W // 2
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
        )
        dist = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)

        low_mask = (dist <= low_r).float()
        high_mask = (dist > high_r).float()
        mid_band_mask = ((dist >= low_r) & (dist <= high_r)).float()

        def _ifft(mask):
            return torch.fft.ifft2(F * mask).real

        low_spatial = _ifft(low_mask)
        high_spatial = _ifft(high_mask)
        band_spatial = _ifft(mid_band_mask)

        if mode == "band+high":
            y = band_spatial + high_spatial
        elif mode == "band":
            y = band_spatial
        elif mode == "high":
            y = high_spatial
        elif mode == "low":
            y = low_spatial
        else:
            y = x

        return y.to(dtype)

    for bk, lst in tqdm(deltas_by_key.items()):
        filtered = []
        for delta in lst:
            if delta.ndim == 2:
                filtered.append(_fft_filter_2d(delta))
            else:
                filtered.append(delta)

        if merging_type == "mean":
            merged[bk] = torch.stack(filtered, dim=0).mean(dim=0)
        else:
            acc = filtered[0].clone()
            for t in range(1, len(filtered)):
                acc.add_(filtered[t])
            merged[bk] = acc
    end_time = time.time()
    # print some info about the merging process
    print(
        f"FR merging with merging_type={merging_type}, low_ratio={low_ratio}, "
        f"high_ratio={high_ratio}, mode={mode}, scaling_factor={scaling_factor} "
        f"took {end_time - start_time:.2f} seconds."
    )
    if cache_path is not None and not profile:
        print(f"Caching merged TV to {cache_path}...")
        # Save safetensor
        save_file(merged, cache_path)
    for k in merged.keys():
        merged[k] = merged[k] * scaling_factor

    return merged

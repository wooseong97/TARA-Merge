# Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Directional Anisotropy [CVPR 2026]

**Official PyTorch implementation of [*Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Directional Anisotropy*](https://arxiv.org/abs/2603.26299) [CVPR 2026].**

Wooseong Jeong\* & Wonyoung Lee\* & Kuk-Jin Yoon, Korea Advanced Institute of Science and Technology (KAIST)

Merging multiple Low-Rank Adaptation (LoRA) modules into a single model is a promising approach for constructing general-purpose systems, but it remains challenging because low-rank update directions introduced by LoRA adapters often span different subspaces and contribute unevenly across directions. When merged naively, such mismatches can weaken the directions most critical to certain task losses while overemphasizing relatively less important ones, ultimately reducing the model’s ability to represent all tasks faithfully. We revisit this problem through two perspectives: subspace coverage, which captures how broadly LoRA directions cover diverse representational directions, and anisotropy, which reflects the imbalance of influence across those directions. We then propose TARA-Merging, short for Task-Rank Anisotropy Alignment. It explicitly incorporates task preferences by aligning the merging weights with a preference-weighted cross-entropy pseudo loss with preserving LoRA directions that encode task-relevant subspaces. This alignment ensures that the merged model maintains broad subspace coverage and accounts for anisotropy via direction-wise reweighting. Across eight vision and six NLI benchmarks, TARA-Merging consistently outperforms vanilla and LoRA-aware baselines, demonstrating strong robustness and generalization, and highlighting the importance of addressing both subspace coverage and anisotropy in LoRA merging.

---

## Setup

### 1. Environment

Create the conda environment from the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate tara
```

### 2. LoRA adapter checkpoints

The eight vision LoRA adapters (CLIP ViT-B/32, rank 16) released by the KnOTS authors can be pulled from the Hugging Face Hub and converted in-place using the helper script:

```bash
bash download_ckpts.sh
```

This populates `checkpoints/<dataset>_r16/adapter/` for each of the eight vision tasks (`stanford_cars`, `dtd`, `eurosat`, `gtsrb`, `mnist`, `resisc45`, `sun397`, `svhn`). The six NLI adapters (`mnli`, `qnli`, `snli`, `rte`, `sick`, `scitail`) for LLaMA-3 8B should be placed under `checkpoints/<dataset>_r16/adapter/` following the same layout.

### 3. CLIP classification heads

The vision merging pipeline needs per-task CLIP heads. **These are not bundled with this repo — please build them yourself** following the recipe in the [KnOTs repository](https://github.com/gstoica27/knots) and place the resulting `.pt` files so that the layout matches `cliphead_dir` / `cliphead_name` in each adapter's `config.yaml`. With the default configs that is:

```
checkpoints/clipheads/ViT-B-32-CLIP/
├── stanford_cars_head.pt
├── dtd_head.pt
├── eurosat_head.pt
├── gtsrb_head.pt
├── mnist_head.pt
├── resisc45_head.pt
├── sun397_head.pt
└── svhn_head.pt
```

If you prefer a different directory (e.g. `checkpoints/cliphead/`), update `model.cliphead_dir` in the adapter's `config.yaml` accordingly.

### 4. Datasets

Datasets are **not** downloaded automatically. Prepare them yourself following the splits/scripts described in the [task_vectors repository](https://github.com/mlfoundations/task_vectors) (see also the discussion in their [issue #1](https://github.com/mlfoundations/task_vectors/issues/1) for dataset acquisition tips), and put them under `data/`. With the default configs the vision layout is:

```
data/8vision/
├── stanford_cars/
├── dtd/
├── eurosat/
├── gtsrb/
├── mnist/
├── resisc45/NWPU-RESISC45-split/
├── sun397/
└── svhn/
```

The expected path for each task is given by `data.data_path` in the corresponding `checkpoints/<task>_r16/config.yaml` — adjust it if your local layout differs.

### 5. Per-adapter `config.yaml`

Every `checkpoints/<task>_r16/config.yaml` declares the backbone, task type, data path, CLIP head location, and dataloader settings used both for evaluation and for the merging routine. Inspect / edit these files if you change checkpoint or data locations.

---

## Running the experiments

All merging entry points are exposed as bash functions in `experiments/merging/functions.sh`. Source the file from the repo root, then call the function for the benchmark / method you want to run. Extra CLI flags are forwarded to `src/merge_with_preference.py`.

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
source experiments/merging/functions.sh
export WANDB_MODE=disabled # disable Weights & Biases logging (optional)
```

### Eight vision benchmarks (CLIP ViT-B/32, LoRA rank 16) — Table 1

```bash
# AdaMerging baseline
eight_vision_adamerging

# TARA-Merging (Variant A)
eight_vision_tara_a

# TARA-Merging (Variant B)
eight_vision_tara_b
```

---

## Contact

Wooseong Jeong: stk14570@kaist.ac.kr

## Reference

If you find this code useful, please cite the following paper:

```bibtex
@article{jeong2026preference,
  title={Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Directional Anisotropy},
  author={Jeong, Wooseong and Lee, Wonyoung and Yoon, Kuk-Jin},
  journal={arXiv preprint arXiv:2603.26299},
  year={2026}
}
```

## Acknowledgement

This repository builds on the open-source efforts of several prior works. We especially thank the authors of [KnOTs](https://github.com/gstoica27/knots), and [Task Vectors](https://github.com/mlfoundations/task_vectors) project.

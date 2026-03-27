# Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Directional Anisotropy [CVPR 2026]

**Official PyTorch implementation of [*Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Directional Anisotropy*] [CVPR 2026].**

Wooseong Jeong* & Wonyoung Lee* & Kuk-Jin Yoon, Korea Advanced Institute of Science and Technology (KAIST)

Merging multiple Low-Rank Adaptation (LoRA) modules into a single model is a promising approach for constructing general-purpose systems, but it remains challenging because low-rank update directions introduced by LoRA adapters often span different subspaces and contribute unevenly across directions. When merged naively, such mismatches can weaken the directions most critical to certain task losses while overemphasizing relatively less important ones, ultimately reducing the model’s ability to represent all tasks faithfully. We revisit this problem through two perspectives: subspace coverage, which captures how broadly LoRA directions cover diverse representational directions, and anisotropy, which reflects the imbalance of influence across those directions. We then propose TARA-Merging, short for Task-Rank Anisotropy Alignment. It explicitly incorporates task preferences by aligning the merging weights with a preference-weighted cross-entropy pseudo loss with preserving LoRA directions that encode task-relevant subspaces. This alignment ensures that the merged model maintains broad subspace coverage and accounts for anisotropy via direction-wise reweighting. Across eight vision and six NLI benchmarks, TARA-Merging consistently outperforms vanilla and LoRA-aware baselines, demonstrating strong robustness and generalization, and highlighting the importance of addressing both subspace coverage and anisotropy in LoRA merging.

## We are preparing the code for public release. It will be available here soon.

## Contact
Wooseong Jeong: stk14570@kaist.ac.kr

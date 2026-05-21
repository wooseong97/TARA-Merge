#!/bin/bash

source experiments/merging/backbone.sh

################################################################################
# NLI (LLaMA-3 8B, LoRA rank 16) — TARA Table 2
################################################################################
function six_nli_adamerging {
  backbone_six_nli --merge_type "weight" --loss_type "default" $@
}
function six_nli_tara_a {
  backbone_six_nli --merge_type "tara-a" $@
}
function six_nli_tara_b {
  backbone_six_nli --merge_type "tara-b" $@
}

################################################################################
# Eight vision (CLIP ViT-B/32, LoRA rank 16) — TARA Table 1
################################################################################
function eight_vision_adamerging {
  backbone_eight_vision --merge_type "weight" --loss_type "default" $@
}
function eight_vision_tara_a {
  backbone_eight_vision --merge_type "tara-a" $@
}
function eight_vision_tara_b {
  backbone_eight_vision --merge_type "tara-b" $@
}

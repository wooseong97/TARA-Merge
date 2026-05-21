#!/bin/bash

SIX_NLI_DIR=(
  checkpoints/mnli_r16
  checkpoints/qnli_r16
  checkpoints/snli_r16
  checkpoints/rte_r16
  checkpoints/sick_r16
  checkpoints/scitail_r16
)

EIGHT_VISION_DIR=(
  checkpoints/stanford_cars_r16
  checkpoints/dtd_r16
  checkpoints/eurosat_r16
  checkpoints/gtsrb_r16
  checkpoints/mnist_r16
  checkpoints/resisc45_r16
  checkpoints/sun397_r16
  checkpoints/svhn_r16
)

function backbone_six_nli {
  python src/merge_with_preference.py \
    --run_name "six_nli" \
    --model_dirs ${SIX_NLI_DIR[@]} \
    --ada \
    --lamda 2.4 \
    --lambda_reg 0.0 \
    --num_epochs 10 \
    --batch_size 16 \
    $@
}
function backbone_eight_vision {
  python src/merge_with_preference.py \
    --run_name "eight_vision" \
    --model_dirs ${EIGHT_VISION_DIR[@]} \
    --ada \
    --lamda 3.2 \
    --lambda_reg 0.0 \
    --num_epochs 5 \
    --batch_size 16 \
    $@
}

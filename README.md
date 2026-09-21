# DINOv1-from-scratch

This repository is a from-scratch implementation of DINOv1 for learning and
research purposes.

## Reference results

Official DINO checkpoints, ImageNet-1k validation, top-1 accuracy:

| Architecture | Params | k-NN  | Linear |
|--------------|--------|-------|--------|
| ViT-S/16     | 21M    | 74.5% | 77.0%  |
| ViT-S/8      | 21M    | 78.3% | 79.7%  |
| ViT-B/16     | 85M    | 76.1% | 78.2%  |
| ViT-B/8      | 85M    | 77.4% | 80.1%  |
| ResNet-50    | 23M    | 67.5% | 75.3%  |

Source: [facebookresearch/dino](https://github.com/facebookresearch/dino)

## Targets

### Stage 1 — Evaluation (no training)

Load the official pretrained weights into this repository's own model
definitions and reproduce the reference numbers with this repository's own
k-NN and linear evaluation code. This validates the architecture,
positional-embedding interpolation, and evaluation protocol independently
of training.

| Architecture | Params | k-NN  | Linear |
|--------------|--------|-------|--------|
| ViT-S/16     | 21M    | 74.5% | 77.0%  |
| ViT-S/8      | 21M    | 78.3% | 79.7%  |
| ViT-B/16     | 85M    | 76.1% | 78.2%  |
| ViT-B/8      | 85M    | 77.4% | 80.1%  |
| ResNet-50    | 23M    | 67.5% | 75.3%  |

Passing criterion: within ±0.3% of the reference.

### Stage 2 — Pretraining

Train from random initialisation with this repository's implementation and
reproduce the reference numbers.

| Run              | Epochs | k-NN  | Linear | Reference cost      |
|------------------|--------|-------|--------|---------------------|
| ViT-S/16 (short) | 100    | 69.3% | 74.0%  | 1.75 days, 8 GPUs   |
| ViT-S/16 (short) | 300    | 73.3% | 76.0%  | 2.6 days, 16 GPUs   |
| ViT-S/16 (full)  | 800    | 74.5% | 77.0%  | —                   |
| ResNet-50        | 800    | 67.5% | 75.3%  | —                   |

The 100-epoch ViT-S/16 run is the primary target: the authors publish the
training and linear-evaluation logs for it, so the loss curve can be
compared step by step, not just the final accuracy.

### Qualitative targets

- k-NN and linear accuracy stay close (reference: 74.5% vs 77.0%) — this
  near-parity is specific to DINO with ViT backbones.
- Self-attention maps of the last block segment objects without supervision.
- Ablations behave as reported: removing the momentum encoder collapses
  training; centering without sharpening, or sharpening without centering,
  also collapses.

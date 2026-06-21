# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2026-06-21

### Added
- **VQ-RWKV-7 Model**: Added a VQ-VAE based vision backbone (`VQ_RWKV7` and builder `create_vq_rwkv7`) under `spixrwkv7/models/vq_rwkv7.py` as an ablation baseline.
- **VQTokenizer & VectorQuantizer**: Supported discrete codebook tokenization with straight-through gradient estimation, EMA codebook updates, and automatic `_last_q_loss` calculation.
- **Argparse & Training Script Updates**: Added `--model-type` parameter ("spix" | "vq") across `compare_architectures.py`, `demo.py`, `train.py`, `fast_test_training.py`, `diagnose_training.py`, and `sanity.py`, automatically adding VQ loss to the objective function when `vq` mode is selected.
- **Model Configs**: Registered VQ configuration parameters (codebook_size, downsample_factor) in `configs/model/`.
- **Attention Residuals (AttnRes)**: Implemented depth-wise attention residuals (`block_attn_res`) for both standard and optimized blocks, replacing fixed additive residuals with learned softmax attention over preceding layer/block representations.
- **AttnRes Gating Options**: Supported `"bias"`, `"sigmoid_scalar"`, `"sigmoid_vector"`, and `"learnable_alpha"` gating configurations for the AttnRes mixing step.
- **AttnRes History Modes**: Supported `"block"` (block boundary only) and `"full"` sequence tracking.
- **Depth-Aware Feature Heads**: Adapted `ClassificationHead`, `RegressionHead` (HumorDB), and `SegHead` (ADE20K) to selectively attend to the complete backbone sequence history, resolving data dilution and improving training efficiency.
- **Alternative Superpixel Backends**: Added support for `"grid"`, `"slic"`, `"slico"`, and `"lnsnet"` superpixel tokenization backends in `SuperpixelTokenizer`.
- **LNS-Net Integration**: Implemented learnable superpixel segmentation (LNS-Net, CVPR 2021) with support for automated BSDS checkpoint download and weight loading.
- **Architectural Enhancements**: Configurable normalization layers (`norm_layer="layernorm"|"rmsnorm"`) and activation functions (`act_layer="relu2"|"gelu"|"silu"|"swiglu"`).
- **Register Tokens**: DINOv2-style learnable register tokens (`register_tokens=N`) for global context accumulation.
- **Dynamic Image Scaling**: Support for flexible resolution heights (`img_size=-1` / `img_size>0`).
- **VQ-RWKV-7 Model**: Added a VQ-VAE based vision backbone (`VQ_RWKV7` and builder `create_vq_rwkv7`) under `spixrwkv7/models/vq_rwkv7.py` as an ablation baseline
- **VQTokenizer & VectorQuantizer**: Supported discrete codebook tokenization with straight-through gradient estimation, EMA codebook updates, and automatic `_last_q_loss` calculation
- **Argparse & Training Script Updates**: Added `--model-type` parameter ("spix" | "vq") across `compare_architectures.py`, `demo.py`, `train.py`, `fast_test_training.py`, `diagnose_training.py`, and `sanity.py`, automatically adding VQ loss to the objective function when `vq` mode is selected
- **Model Configs**: Registered VQ configuration parameters (codebook_size, downsample_factor) in `configs/model/`

### Changed
- **Type Checker Cleanups**: Cleaned up dynamic attribute checks in script tasks to use `getattr` and `setattr` to guarantee complete Pyright compatibility.
- **Modular Refactoring**: Reorganized the architecture into separate modular classes: `RecurrentScan`, `SpatialMixer`, `ChannelMix`, `SuperpixelTokenizer`, `_DynamicOffset`, and `_TimeMixParams`.
- **Project Structure**: Relocated the core backbone definition file from `spixrwkv7/spixrwkv7.py` to `spixrwkv7/models/spixrwkv7.py`.
- **Inference & Demo Updates**: Expose `--use-attnres` option in `scripts/demo.py` and diagnostic training script parameters.
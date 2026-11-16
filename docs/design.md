# Design decisions

The decisions this runtime is built on, stated here so the repository can be read on its own.

| Id | Decision | What it means here |
|---|---|---|
| D7 | New components are written in C++ (C++23 today, moving to C++26) | The runtime, its backends and its tools are C++; no other language is on the build or run path |
| D16 | A component is a library first, and ships its own tests | The artifact is the `orama::model-runtime` library; `gemma-cli` and `gemma-bench` are thin tools over it, and `ctest` runs its own test binaries |
| D19 | Training and inference sit behind one backend interface, with a native backend per platform | MLX on macOS first; libtorch or llama.cpp on Linux and Windows; CUDA later. Callers use the interface and never a backend directly |
| D21 | Modules are small and single-responsibility; there is no catch-all "common" module | This repository does backend-agnostic train and infer and nothing else. The models that run on it live with the components that use them |
| D26 | The model path is pure C++ on `mlx::core`, with no Python anywhere | Weight loading, the tokenizer (SentencePiece, a C++ library), the forward pass, the decode loop and LoRA fine-tuning are C++ against `mlx::core`. Python `mlx_lm` is read as a reference for porting model definitions, and is never a runtime or build dependency |

D26 is feasible because MLX ships the heavy kernels as C++ ops: `mx::fast::rms_norm`, `mx::fast::rope`
and `mx::fast::scaled_dot_product_attention`; `mx::load_safetensors` and `mx::load_gguf` for weights;
`mx::quantized_matmul` and `mx::dequantize` for 4-bit weights; and `mx::value_and_grad` for the
gradients LoRA needs.

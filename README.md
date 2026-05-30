# orama-model-runtime

**Backend-agnostic model train + infer** for the [Orama Platform](https://github.com/oramasearch/orama-platform), behind **one interface** (architecture **D19**).

The same training/inference contract runs on a **native backend per platform** — **MLX** on macOS, **libtorch** or **llama.cpp** on Linux/Windows, **CUDA** in the cloud. **Not MLX-specific**: MLX is the first target, but the interface is there from day one so each environment uses its native engine. The specific models (the two Gemma-4 NL→orly / orly→tools models) live in the `policy` component and run *on* this runtime.

Small, single-responsibility module ([D21](https://github.com/oramasearch/orama-platform/blob/main/docs/projects/orama-platform/architecture.md#d21)). Design + planning:
- Decision: [D19](https://github.com/oramasearch/orama-platform/blob/main/docs/projects/orama-platform/architecture.md#d19) · the deep "why": [`experience-river.md`](https://github.com/oramasearch/orama-platform/blob/main/docs/projects/orama-platform/experience-river.md)
- Worked example: [`research/mlx-finetuning-gemma4-lora.md`](https://github.com/oramasearch/orama-platform/blob/main/docs/projects/orama-platform/research/mlx-finetuning-gemma4-lora.md)
- Tracking: [`#33`](https://github.com/oramasearch/orama-platform/issues/33) / [`#34`](https://github.com/oramasearch/orama-platform/issues/34) (the models) · [`#35`](https://github.com/oramasearch/orama-platform/issues/35) / [`#36`](https://github.com/oramasearch/orama-platform/issues/36) (train/eval)

## Build

```sh
cmake -S . -B build -G Ninja && cmake --build build   # interface + tests, no backend
ctest --test-dir build --output-on-failure
```

Backends are opt-in CMake toggles (`-DORAMA_BACKEND_MLX=ON`, `…_LIBTORCH`, `…_LLAMACPP`, `…_CUDA`) so the scaffold + CI build with no ML toolchain.

## Stack

C++23 (→ C++26 per D7), CMake ≥ 3.28 + Ninja, FetchContent, Catch2. Library-first (D16); links as `orama::model-runtime`.

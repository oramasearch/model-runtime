# orama-model-runtime — Claude rules

**Backend-agnostic model train + infer** for the [Orama Platform](https://github.com/oramasearch/orama-platform). Scaffold stage. **Code only** — design lives in the platform repo.

## What this is

ONE interface for training + inference (D19), with a **native backend per platform**: **MLX** (macOS, first target) · **libtorch** / **llama.cpp** (Linux/Windows) · **CUDA** (cloud). **This module is the abstraction + the backends — NOT MLX-specific.** Multiple AI backends are a first-class requirement; never hard-code MLX above the backend seam.

## Single responsibility (D21)

**Only** the train/infer backend abstraction + the backends. It does **not** own:
- the specific models / weights / fine-tune orchestration → that's the `policy` component (it runs *on* this runtime),
- the language those models target → `orly`,
- the experience/reward loop that produces training data → the experience-rig.

Keep this module about *running a model on whatever hardware we're on*, nothing more.

## Runbook

Build/run/validate/benchmark the **real Gemma‑2** model (pure C++/MLX, no Python): **[`docs/gemma-runbook.md`](docs/gemma-runbook.md)** — weights download, `gemma-cli`, the `sudo gemma-bench` GPU perf harness (tok/s + active-residency/power/freq), the garbage-if-wrong checklist, troubleshooting. Measured baseline on M3 Max: prefill ~770 tok/s, decode ~95 tok/s, **99% GPU residency**, ~21.8 W.

## Where the design lives

- architecture [D19](https://github.com/oramasearch/orama-platform/blob/main/docs/projects/orama-platform/architecture.md#d19) / [D21](https://github.com/oramasearch/orama-platform/blob/main/docs/projects/orama-platform/architecture.md#d21); [`experience-river.md`](https://github.com/oramasearch/orama-platform/blob/main/docs/projects/orama-platform/experience-river.md).
- Worked example: [`research/mlx-finetuning-gemma4-lora.md`](https://github.com/oramasearch/orama-platform/blob/main/docs/projects/orama-platform/research/mlx-finetuning-gemma4-lora.md).
- Tracking: [`#33`](https://github.com/oramasearch/orama-platform/issues/33)/[`#34`](https://github.com/oramasearch/orama-platform/issues/34)/[`#35`](https://github.com/oramasearch/orama-platform/issues/35)/[`#36`](https://github.com/oramasearch/orama-platform/issues/36).

## Stack + conventions

- **C++23** today, **C++26** target (D7). CMake ≥ 3.28 + Ninja; FetchContent; Catch2. Library-first (D16) — links as `orama::model-runtime`.
- **Backends are opt-in CMake toggles** (`ORAMA_BACKEND_{MLX,LIBTORCH,LLAMACPP,CUDA}`), all OFF by default so the interface + tests build with no ML toolchain.
- Per-module error model here if needed — **no shared `common` module** (D21).
- **Conventional Commits**; **never mention "Claude"/"AI"/"Codex"** (org policy). Private repo — push to `main` directly.

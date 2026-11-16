# orama-model-runtime — Claude rules

**Backend-agnostic model train + infer** for Orama Lake. Scaffold stage. Read `README.md` and
[`docs/design.md`](docs/design.md) (the design decisions D7, D16, D19, D21, D26) first.

## What this is

ONE interface for training + inference (D19), with a **native backend per platform**: **MLX** (macOS, first target) · **libtorch** / **llama.cpp** (Linux/Windows) · **CUDA** (cloud). **This module is the abstraction + the backends — NOT MLX-specific.** Multiple AI backends are a first-class requirement; never hard-code MLX above the backend interface.

## Single responsibility (D21)

**Only** the train/infer backend abstraction + the backends. It does **not** own:
- the specific models / weights / fine-tune orchestration → the components that use them (they run *on* this runtime),
- the language those models target,
- the experience/reward loop that produces training data.

Keep this module about *running a model on whatever hardware we're on*, nothing more.

## Build and test

```sh
cmake -S . -B build -G Ninja && cmake --build build   # interface + tests, no backend
ctest --test-dir build --output-on-failure
```

Real Gemma‑2 (pure C++/MLX, no Python): **[`docs/gemma-runbook.md`](docs/gemma-runbook.md)** — weights download, `gemma-cli`, the `sudo gemma-bench` GPU perf harness (tok/s + active-residency/power/freq), the garbage-if-wrong checklist, troubleshooting.

## Stack + conventions

- **C++23** today, **C++26** target (D7). CMake ≥ 3.28 + Ninja; FetchContent; Catch2. Library-first (D16) — links as `orama::model-runtime`.
- **Backends are opt-in CMake toggles** (`ORAMA_BACKEND_{MLX,LIBTORCH,LLAMACPP,CUDA}`), all OFF by default so the interface + tests build with no ML toolchain.
- Per-module error model here if needed — **no shared `common` module** (D21).
- Docs in the same commit as the code (`README.md`, `docs/`).
- **Conventional Commits**; feature branch + pull request, never a push to the default branch.
- **No AI attribution anywhere**: never an AI `Co-Authored-By` trailer, "Generated with", or a mention of an AI in commits, pull requests, tags or release notes.
- No time or size estimates.
- Never write the word "seam": name the mechanism (interface, extension point, boundary, injected dependency).

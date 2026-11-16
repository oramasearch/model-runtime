# orama-model-runtime

**Backend-agnostic model train + infer** for Orama Lake, behind **one interface** (design decision
[D19](docs/design.md)).

The same training/inference contract runs on a **native backend per platform** — **MLX** on macOS, **libtorch** or **llama.cpp** on Linux/Windows, **CUDA** in the cloud. **Not MLX-specific**: MLX is the first target, but the interface is there from day one so each environment uses its native engine. The specific models live with the components that use them and run *on* this runtime.

Small, single-responsibility module ([D21](docs/design.md)). The decisions it is built on (D7, D16,
D19, D21, D26) are in [`docs/design.md`](docs/design.md).

Released: [v0.1.0](docs/releases/v0.1.0.md) ([all releases](docs/releases/README.md)).

## Build

```sh
cmake -S . -B build -G Ninja && cmake --build build   # interface + tests, no backend
ctest --test-dir build --output-on-failure
```

Backends are opt-in CMake toggles (`-DORAMA_BACKEND_MLX=ON`, `…_LIBTORCH`, `…_LLAMACPP`, `…_CUDA`) so the scaffold + CI build with no ML toolchain.

**Real Gemma‑2 (pure C++/MLX, no Python):** build/run/validate/benchmark the real model — see **[`docs/gemma-runbook.md`](docs/gemma-runbook.md)** (weights download, `gemma-cli`, the `sudo gemma-bench` perf harness, the garbage-if-wrong checklist, troubleshooting).

## Stack

C++23 (→ C++26 per [D7](docs/design.md)), CMake ≥ 3.28 + Ninja, FetchContent, Catch2. Library-first
([D16](docs/design.md)); links as `orama::model-runtime`.
C++23 (→ C++26 per D7), CMake ≥ 3.28 + Ninja, FetchContent, Catch2. Library-first (D16); links as `orama::model-runtime`.

## License

MIT, see [LICENSE](LICENSE). Third-party components and their licenses are in
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES). Contributions are welcome under the
[Contributor License Agreement](CLA.md); see [CONTRIBUTING.md](CONTRIBUTING.md).

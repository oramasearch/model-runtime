# Runbook — real Gemma‑2 on MLX (pure C++, no Python)

Operational guide to build, run, validate, and benchmark the **real Gemma‑2
language model** in this repo. Everything is **pure C++ on `mlx::core`** — no
Python, no `mlx_lm` ([D26](design.md)).
Implementation: [`src/gemma.cpp`](../src/gemma.cpp) + [`include/model/gemma.hpp`](../include/model/gemma.hpp);
tokenizer [`src/tokenizer.cpp`](../src/tokenizer.cpp); weights [`src/weights.cpp`](../src/weights.cpp).

> Target: **macOS, Apple Silicon (arm64)**. Validated on an **M3 Max**.

---

## 0. Prerequisites (one-time)

```sh
brew install cmake ninja mlx sentencepiece
```
- **`mlx`** is a prebuilt bottle (Metal compiled in) → **you do NOT need full Xcode**, only Command Line Tools.
- **`sentencepiece`** is the C++ tokenizer lib (not the Python package).
- No CUDA, no Python, no model-download tool.

---

## 1. Build (MLX backend on)

```sh
cd modules/model-runtime
cmake -S . -B build-mlx -G Ninja -DORAMA_BACKEND_MLX=ON
cmake --build build-mlx
ctest --test-dir build-mlx          # 15/15 (round-trip + real-MLX LoRA + tokenizer/weights)
```
Produces `build-mlx/gemma-cli` (generate) and `build-mlx/gemma-bench` (perf).

> ⚠️ `build-mlx/` is **git-ignored** — it does not survive a fresh checkout.
> Re-run the two `cmake` lines above before running the binaries (this is what
> bites you with `sudo ./build-mlx/gemma-bench: command not found`).

Without `-DORAMA_BACKEND_MLX=ON` the module builds the deterministic **stub**
engine only (no ML toolchain needed) — that's the default for CI and the
no-Gemma desktop build.

---

## 2. Get the weights (data download — no Python)

`mlx-community/gemma-2-2b-it-4bit` is **public** (not gated). The weights are not part of this
repository: you download them yourself, and their use is under Google's
[Gemma Terms of Use](https://ai.google.dev/gemma/terms). Fetch the files straight over HTTPS:

```sh
mkdir -p ~/models/gemma-2-2b-it-4bit && cd ~/models/gemma-2-2b-it-4bit
B=https://huggingface.co/mlx-community/gemma-2-2b-it-4bit/resolve/main
for f in config.json tokenizer.model model.safetensors; do curl -L "$B/$f" -o "$f"; done
```
~1.5 GB. Only `model.safetensors` + `tokenizer.model` are read at runtime
(`config.json` is informational — the architecture is baked into `GemmaConfig`
for this exact model). Other Gemma‑2 sizes need their constants filled in.

**License.** The weights are not part of this repository and are not distributed with it. Gemma is
provided under and subject to the Gemma Terms of Use (https://ai.google.dev/gemma/terms), including
its Prohibited Use Policy; by downloading the weights you accept those terms. This repository's own
code, an independent implementation of the Gemma 2 architecture, is MIT (see `LICENSE`).

---

## 3. Validate generation

```sh
./build-mlx/gemma-cli ~/models/gemma-2-2b-it-4bit "The capital of France is"
./build-mlx/gemma-cli ~/models/gemma-2-2b-it-4bit "Explain what BM25 is in one sentence."
```
**Expected:** coherent, correct output — e.g. *“The capital of France is **Paris**. 🇫🇷”* and a correct BM25 definition. A wrong forward (bad RoPE, missing softcap, wrong RMSNorm/quant layout) produces garbage or repeated whitespace, so a sensible answer **is** the validation.

> Greedy decoding **repeats** after the coherent opening — that's expected (no sampler yet), not a correctness bug.

Debug a single run (per-token id + logit range):
```sh
GEMMA_DEBUG=1 ./build-mlx/gemma-cli ~/models/gemma-2-2b-it-4bit "The capital of France is"
```

---

## 4. Benchmark (tokens/sec + GPU stats)

`gemma-bench` spawns macOS **`powermetrics`** (GPU sampler, needs root) and a
generation workload concurrently, then prints one report. **Run with `sudo`**
(in a real Terminal, so the password prompt works):

```sh
sudo ./build-mlx/gemma-bench ~/models/gemma-2-2b-it-4bit 8     # 8 = sampling seconds
```

**Measured baseline — M3 Max, gemma‑2‑2b‑it‑4bit, pure C++/MLX:**

| metric | value |
|---|---|
| prefill | **~770 tok/s** (short prompt) · **~1,670 tok/s** (≈400‑tok prompt, parallel) |
| decode | **~95–98 tok/s** (short ctx) · **~61 tok/s** (≈400‑tok ctx — grows with KV length) |
| peak GPU memory | **~2.9 GB** (short) · **~4.5 GB** (long prompt) |
| **GPU active residency** | **~99 %** — fully GPU-bound / saturated |
| GPU power | **~21.8 W** |
| GPU frequency | **~1,377 MHz** |

Without `sudo`, `gemma-bench` still reports tok/s + peak memory and prints
`GPU samples (skipped — run with sudo)`. For a no-sudo live view, open
**Activity Monitor → Window → GPU History (⌘4)** while it runs.

Per-run timing without the harness:
```sh
GEMMA_STATS=1 ./build-mlx/gemma-cli ~/models/gemma-2-2b-it-4bit "your prompt"
# stderr: prefill tok/s | decode tok/s | peak GPU mem
```

---

## 5. Use it from the agent / desktop

The `MlxEngine` lazy-loads Gemma when `ORAMA_GEMMA_DIR` is set, so the agent and
the desktop generate real Gemma text:

```sh
export ORAMA_GEMMA_DIR=~/models/gemma-2-2b-it-4bit
# build the desktop with the MLX backend (propagates through agent → model-runtime):
cd ../installer
cmake -S . -B build-mlx -G Ninja -DORAMA_INSTALLER_GUI=ON -DORAMA_BACKEND_MLX=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-mcpu=native"
cmake --build build-mlx && ./build-mlx/orama-desktop
```
The Agent tab's model line then shows `mlx (real Gemma-2 @ Metal)`.

---

## 6. Architecture notes (the garbage-if-wrong list)

Gemma‑2 has several quirks; each is implemented in `gemma.cpp` and each yields
broken output if dropped:
- **4‑bit quantized linears + tied quantized embedding** — `mx::quantized_matmul(..., transpose=true, group_size=64, bits=4)`; logits reuse `embed_tokens` (no `lm_head`).
- **Sandwich RMSNorm, 4 norms/layer**, each `(1 + weight)`: `input_` + `post_attention_` (around attn), `pre_feedforward_` + `post_feedforward_` (around MLP).
- **Embedding scaled by √hidden** (= √2304 = 48.0).
- **Softcapping**: attention logits `50·tanh(x/50)` (so we do attention **manually**, not `fast::scaled_dot_product_attention`); final logits `30·tanh(x/30)`.
- **GQA** 8 heads / 4 KV (repeat 2×); **head_dim 256** (≠ hidden/heads); attn scale `1/√query_pre_attn_scalar = 1/16`.
- **`gelu_pytorch_tanh`** activation in the SwiGLU MLP.
- **RoPE** `mx::fast::rope(dims=256, traditional=false, base=10000, offset=pos)`.

Full rationale: [D26 and D19](design.md). Reference implementations in Python (MLX-LM) are read
only to port the *architecture*; their `nn::Module`/`AdamW` are Python and are not used.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `sudo ./build-mlx/gemma-bench: command not found` | `build-mlx/` is gitignored — rebuild (step 1). |
| Garbage / all-whitespace output | a Gemma quirk (§6) is wrong — check softcaps, RMSNorm `1+w`, embed scale, quant `transpose/group_size/bits`. |
| GPU stat lines empty after sudo | `powermetrics` output format varies by macOS; the parser looks for `active residency` / `GPU Power` / `active frequency`. Capture `sudo powermetrics --samplers gpu_power -n 1` and adjust `tools/gemma_bench.cpp:num_after`. |
| `Could NOT find MLX` / sentencepiece | `brew install mlx sentencepiece`; the CMake hints `/opt/homebrew/opt/{mlx,sentencepiece}`. |
| Decode slower than expected | it scales with context length (KV cache grows); short prompts ≈ 95 tok/s, long ≈ 60. |

---

## 8. Known limitations / next

- **Greedy only** — repeats after the opening; a sampler (temperature / top‑p / repetition penalty) is the next add (kills repetition).
- **LoRA is the toy net** in `MlxEngine::train_step`; LoRA over the *real* Gemma projections is the remaining work.
- **Sliding-window attention** is approximated as full causal (exact for context < 4096; Gemma‑2's per-layer sliding window only matters beyond that).
- **Single baked config** (gemma‑2‑2b‑it); other sizes/families need their `GemmaConfig` constants.

#pragma once
// model::Engine — backend-agnostic train + infer behind ONE interface (D19).
//
// infer() is a TOKEN-STREAMING, STEERABLE decode (D25 "active decoding"): the
// engine emits tokens one at a time through `on_token`, and the callback may
// return text to **inject into the context before the next token** — this is
// how the agent weaves search / tool calls into the decode loop (the model
// "searches as it decodes"). The stub backend makes the loop real even without
// a model; the MLX backend (macOS/Metal, behind ORAMA_BACKEND_MLX) swaps in a
// real small model later.
//
// No exceptions cross the API (D7/D16) — methods return Result<T>.
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "model/backend.hpp"
#include "model/error.hpp"

namespace model {

// Per-token hook. Receives the token just produced; may RETURN text to inject
// into the context before the next token (empty = inject nothing). This is the
// D25 seam: the agent runs a data-node search here and feeds results back in.
using OnToken = std::function<std::string(std::string_view token)>;

struct InferResult {
    std::string text;            // full decoded text (tokens + injections, in order)
    std::size_t tokens = 0;      // tokens the model emitted (excludes injected text)
    std::size_t injections = 0;  // times the callback injected context mid-decode
    double took_us = 0.0;
};

struct TrainResult {
    double loss = 0.0;           // loss after the step
    std::uint64_t step = 0;      // monotonic step counter
    double took_us = 0.0;
};

// One backend instance. Construct via make_engine().
class Engine {
public:
    virtual ~Engine() = default;

    // e.g. "mlx (stub)" — backend + whether it's the real model or the stub.
    [[nodiscard]] virtual std::string_view name() const = 0;
    [[nodiscard]] virtual Backend backend() const = 0;
    [[nodiscard]] virtual bool is_stub() const = 0;

    // Streaming decode. Emits tokens via on_token; injected returns are folded
    // into the output before the next token. Returns the full result.
    [[nodiscard]] virtual Result<InferResult> infer(std::string_view prompt,
                                                    const OnToken& on_token = {}) = 0;

    // One LoRA-style training step over a few (prompt, target) samples; returns
    // the loss. Proves the on-device train path (D19). Synthetic for the stub.
    [[nodiscard]] virtual Result<TrainResult> train_step(
        const std::vector<std::pair<std::string, std::string>>& samples) = 0;
};

// Build the engine for a backend. Today every backend returns a deterministic
// stub (name suffixed " (stub)"); real backends land behind the CMake toggles.
[[nodiscard]] std::unique_ptr<Engine> make_engine(Backend b = default_backend());

}  // namespace model

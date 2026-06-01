#include "model/engine.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <string>

namespace model {
namespace {

using Clock = std::chrono::steady_clock;
double us_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

// A deterministic, dependency-free stand-in for a real small model. It makes
// the decode loop REAL (token-by-token, steerable) without any weights, so the
// installer → agent → data-node chain + the debug UI work end-to-end today.
// The fixed token stream includes a "[search]" marker so the agent's on_token
// hook can run a data-node query and inject the result mid-decode (D25).
class StubEngine final : public Engine {
public:
    explicit StubEngine(Backend b) : backend_(b), name_(std::string(label(b)) + " (stub)") {}

    std::string_view name() const override { return name_; }
    Backend backend() const override { return backend_; }
    bool is_stub() const override { return true; }

    Result<InferResult> infer(std::string_view prompt, const OnToken& on_token) override {
        const auto t0 = Clock::now();
        // The stub's "decode": a fixed plan that pauses to search, then answers.
        static constexpr std::array<std::string_view, 4> kStream{
            "plan", "[search]", "synthesize", "answer"};
        InferResult r;
        std::string out;
        for (const auto tok : kStream) {
            if (!out.empty()) out += ' ';
            out += tok;
            ++r.tokens;
            if (on_token) {
                const std::string injected = on_token(tok);
                if (!injected.empty()) {
                    out += ' ';
                    out += injected;
                    ++r.injections;
                }
            }
        }
        (void)prompt;  // a real backend conditions on the prompt; the stub is fixed
        r.text = std::move(out);
        r.took_us = us_since(t0);
        return r;
    }

    Result<TrainResult> train_step(
        const std::vector<std::pair<std::string, std::string>>& samples) override {
        const auto t0 = Clock::now();
        ++step_;
        // Synthetic LoRA-step loss: monotonically decreasing, nudged by batch size.
        const double batch = samples.empty() ? 1.0 : static_cast<double>(samples.size());
        TrainResult r;
        r.step = step_;
        r.loss = (2.0 / (1.0 + static_cast<double>(step_))) / (1.0 + 0.05 * batch);
        r.took_us = us_since(t0);
        return r;
    }

private:
    Backend backend_;
    std::string name_;
    std::uint64_t step_ = 0;
};

}  // namespace

std::unique_ptr<Engine> make_engine(Backend b) {
    // Today: always the stub. Real backends (MLX first, D19) land behind the
    // CMake toggles and will be selected here when present.
    return std::make_unique<StubEngine>(b);
}

}  // namespace model

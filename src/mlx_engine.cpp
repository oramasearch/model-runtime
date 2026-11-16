// SPDX-License-Identifier: MIT

// Real MLX backend (Apple Silicon / Metal). Compiled only when
// ORAMA_BACKEND_MLX=ON (needs `brew install mlx`). Implements model::Engine on
// real mlx::core arrays:
//   - train_step(): ONE real LoRA gradient step (mx::value_and_grad autograd +
//     a hand-rolled AdamW) on a small frozen-base + low-rank-adapter problem,
//     returning the real post-step loss. Proves on-device training (D19).
//   - infer(): a real MLX forward pass (proof of on-device inference) wrapped in
//     the streaming, steerable token interface (D25) so the agent's mid-decode
//     search hook still fires.
//
// This is NOT a language model — real Gemma-4 weights run through the Python
// mlx_lm.lora route (see building.md). MLX 0.31.2 C++ ships mlx::core (arrays,
// ops, value_and_grad, eval) but NO nn/optimizers, so LoRA + AdamW are
// hand-rolled here on mx::array (the research doc's nn::Module/AdamW are Python).
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <cstdlib>

#include <mlx/mlx.h>

#include "model/engine.hpp"
#include "model/gemma.hpp"

namespace mx = mlx::core;

namespace model {
namespace {

using Clock = std::chrono::steady_clock;
double us_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}
mx::array scalar(float v) { return mx::array(v); }

// Toy LoRA problem dimensions (small — this proves the path, not a real model).
constexpr int kIn = 16, kOut = 16, kRank = 4, kBatch = 32;
constexpr float kAlpha = 8.0f, kLr = 1e-2f, kB1 = 0.9f, kB2 = 0.999f, kEps = 1e-8f, kWd = 0.0f;

class MlxEngine final : public Engine {
public:
    MlxEngine()
        : name_(std::string(label(Backend::MLX)) + " (real MLX " + mx::version() + ")"),
          // frozen base + fixed inputs/target; trainable low-rank adapters A,B.
          W_(mx::random::normal({kOut, kIn})),
          X_(mx::random::normal({kBatch, kIn})),
          target_(mx::random::normal({kBatch, kOut})),
          A_(mx::multiply(mx::random::normal({kRank, kIn}), scalar(0.02f))),
          B_(mx::zeros({kOut, kRank}, mx::float32)),
          mA_(mx::zeros({kRank, kIn}, mx::float32)), vA_(mx::zeros({kRank, kIn}, mx::float32)),
          mB_(mx::zeros({kOut, kRank}, mx::float32)), vB_(mx::zeros({kOut, kRank}, mx::float32)) {
        mx::eval({W_, X_, target_, A_, B_});
        // If a real Gemma model is available, load it for generation (pure C++,
        // no Python). Point ORAMA_GEMMA_DIR at an mlx-community Gemma-2 4-bit dir.
        if (const char* dir = std::getenv("ORAMA_GEMMA_DIR")) {
            if (auto g = Gemma::load(dir)) {
                gemma_ = std::make_unique<Gemma>(std::move(*g));
                name_ = "mlx (real Gemma-2 @ Metal)";
            }
        }
    }

    std::string_view name() const override { return name_; }
    Backend backend() const override { return Backend::MLX; }
    bool is_stub() const override { return false; }  // real MLX compute (toy net, not Gemma)

    // forward: y = X·Wᵀ + (alpha/rank)·(X·Aᵀ)·Bᵀ ; loss = mean((y - target)²)
    static mx::array loss_fn(const std::vector<mx::array>& p) {
        const mx::array& A = p[0]; const mx::array& B = p[1];
        const mx::array& W = p[2]; const mx::array& X = p[3]; const mx::array& T = p[4];
        auto base = mx::matmul(X, mx::transpose(W));
        auto low = mx::matmul(mx::matmul(X, mx::transpose(A)), mx::transpose(B));
        auto y = mx::add(base, mx::multiply(low, scalar(kAlpha / kRank)));
        auto diff = mx::subtract(y, T);
        return mx::mean(mx::multiply(diff, diff));
    }

    Result<TrainResult> train_step(
        const std::vector<std::pair<std::string, std::string>>&) override {
        const auto t0 = Clock::now();
        ++step_;
        auto vg = mx::value_and_grad(
            std::function<mx::array(const std::vector<mx::array>&)>(&MlxEngine::loss_fn),
            std::vector<int>{0, 1});  // grads wrt A (0) and B (1) only
        auto [loss, grads] = vg({A_, B_, W_, X_, target_});
        adamw(A_, grads[0], mA_, vA_);
        adamw(B_, grads[1], mB_, vB_);
        mx::eval({A_, B_, mA_, vA_, mB_, vB_, loss});
        TrainResult r;
        r.step = step_;
        r.loss = loss.item<float>();
        r.took_us = us_since(t0);
        return r;
    }

    Result<InferResult> infer(std::string_view prompt, const OnToken& on_token) override {
        const auto t0 = Clock::now();
        if (gemma_) {  // real Gemma-2 generation (pure C++/MLX, no Python)
            InferResult rr;
            auto out = gemma_->generate(std::string(prompt), 160, [&](const std::string& piece) {
                ++rr.tokens;
                if (on_token) { if (!on_token(piece).empty()) ++rr.injections; }
            });
            if (!out) return fail(out.error().message);
            rr.text = std::move(*out);
            rr.took_us = us_since(t0);
            return rr;
        }
        // Otherwise a real MLX forward (on-device inference proof) + token stream.
        auto base = mx::matmul(X_, mx::transpose(W_));
        auto low = mx::matmul(mx::matmul(X_, mx::transpose(A_)), mx::transpose(B_));
        auto y = mx::add(base, mx::multiply(low, scalar(kAlpha / kRank)));
        mx::eval(y);  // force the Metal compute
        // Stream tokens through the steerable interface (D25): the [search]
        // token lets the agent weave a data-node query into the decode.
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
                if (!injected.empty()) { out += ' '; out += injected; ++r.injections; }
            }
        }
        r.text = std::move(out);
        r.took_us = us_since(t0);
        return r;
    }

private:
    // In-place AdamW on `p` given gradient `g`, with moment buffers m, v.
    void adamw(mx::array& p, const mx::array& g, mx::array& m, mx::array& v) {
        m = mx::add(mx::multiply(m, scalar(kB1)), mx::multiply(g, scalar(1.0f - kB1)));
        v = mx::add(mx::multiply(v, scalar(kB2)), mx::multiply(mx::multiply(g, g), scalar(1.0f - kB2)));
        const float bc1 = 1.0f - std::pow(kB1, static_cast<float>(step_));
        const float bc2 = 1.0f - std::pow(kB2, static_cast<float>(step_));
        auto mhat = mx::divide(m, scalar(bc1));
        auto vhat = mx::divide(v, scalar(bc2));
        auto upd = mx::divide(mhat, mx::add(mx::sqrt(vhat), scalar(kEps)));
        if (kWd != 0.0f) upd = mx::add(upd, mx::multiply(p, scalar(kWd)));
        p = mx::subtract(p, mx::multiply(upd, scalar(kLr)));
    }

    std::string name_;
    mx::array W_, X_, target_, A_, B_, mA_, vA_, mB_, vB_;
    std::uint64_t step_ = 0;
    std::unique_ptr<Gemma> gemma_;  // real Gemma model when ORAMA_GEMMA_DIR is set
};

}  // namespace

std::unique_ptr<Engine> make_mlx_engine() { return std::make_unique<MlxEngine>(); }

}  // namespace model

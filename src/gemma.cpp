#include "model/gemma.hpp"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "model/tokenizer.hpp"
#include "model/weights.hpp"

namespace mx = mlx::core;
namespace mxf = mlx::core::fast;

namespace model {
namespace {

const mx::Dtype kDt = mx::bfloat16;            // compute dtype (matches the model)
mx::array k(float v) { return mx::array(v, kDt); }  // a scalar in compute dtype

// A 4-bit quantized linear: y = x @ dequant(w)^T  (mlx packs along the last axis).
struct QLinear {
    const mx::array* w = nullptr;
    const mx::array* s = nullptr;
    const mx::array* b = nullptr;
    int gs = 64, bits = 4;
    mx::array operator()(const mx::array& x) const {
        return mx::quantized_matmul(x, *w, *s, *b, /*transpose=*/true, gs, bits);
    }
};

// gelu(x) with the tanh approximation (Gemma's gelu_pytorch_tanh).
mx::array gelu_tanh(const mx::array& x) {
    auto x3 = mx::multiply(mx::multiply(x, x), x);
    auto inner = mx::multiply(mx::add(x, mx::multiply(x3, k(0.044715f))), k(0.7978845608028654f));
    return mx::multiply(mx::multiply(x, k(0.5f)), mx::add(mx::tanh(inner), k(1.0f)));
}

}  // namespace

struct Gemma::Impl {
    GemmaConfig cfg;
    Tokenizer tok;
    Weights w;
    mx::array embed = mx::array(0.0f);   // [vocab, hidden] dequantized, for input lookup
    std::vector<mx::array> norm1p;       // (1 + weight) for every norm, by index (see norm_key)
    std::vector<mx::array> kcache, vcache;

    QLinear ql(const std::string& prefix) const {
        return QLinear{&w.at(prefix + ".weight"), &w.at(prefix + ".scales"),
                       &w.at(prefix + ".biases"), cfg.group_size, cfg.bits};
    }
    // RMSNorm with Gemma's (1 + weight); weight already pre-incremented in `wt`.
    static mx::array rmsnorm(const mx::array& x, const mx::array& wt, float eps) {
        return mxf::rms_norm(x, wt, eps);
    }
};

Gemma::Gemma() : impl_(std::make_unique<Impl>()) {}
Gemma::~Gemma() = default;
Gemma::Gemma(Gemma&&) noexcept = default;
Gemma& Gemma::operator=(Gemma&&) noexcept = default;
const GemmaConfig& Gemma::config() const { return impl_->cfg; }

Result<Gemma> Gemma::load(const std::string& dir) {
    Gemma g;
    auto& im = *g.impl_;
    if (auto r = im.tok.load(dir + "/tokenizer.model"); !r) return fail(r.error().message);
    auto w = Weights::load(dir + "/model.safetensors");
    if (!w) return fail(w.error().message);
    im.w = std::move(*w);
    try {
        // Dequantize the (quantized, tied) embedding once for input lookup.
        im.embed = mx::astype(
            mx::dequantize(im.w.at("model.embed_tokens.weight"), im.w.at("model.embed_tokens.scales"),
                           im.w.at("model.embed_tokens.biases"), im.cfg.group_size, im.cfg.bits),
            kDt);
        mx::eval(im.embed);
    } catch (const std::exception& e) {
        return fail(std::string("gemma load (embed dequant): ") + e.what());
    }
    return g;
}

Result<std::string> Gemma::generate(const std::string& prompt, int max_new_tokens,
                                    const std::function<void(const std::string&)>& on_piece,
                                    GenStats* stats) {
    auto& im = *impl_;
    const auto& c = im.cfg;
    if (!im.tok.loaded()) return fail("tokenizer not loaded");

    try {
        // ---- (1 + weight) for every norm, cached ----
        auto one_plus = [&](const std::string& key) {
            return mx::astype(mx::add(mx::astype(im.w.at(key), kDt), k(1.0f)), kDt);
        };
        std::vector<mx::array> in_ln, post_attn_ln, pre_ffn_ln, post_ffn_ln;
        for (int i = 0; i < c.n_layers; ++i) {
            const std::string p = "model.layers." + std::to_string(i) + ".";
            in_ln.push_back(one_plus(p + "input_layernorm.weight"));
            post_attn_ln.push_back(one_plus(p + "post_attention_layernorm.weight"));
            pre_ffn_ln.push_back(one_plus(p + "pre_feedforward_layernorm.weight"));
            post_ffn_ln.push_back(one_plus(p + "post_feedforward_layernorm.weight"));
        }
        auto final_ln = one_plus("model.norm.weight");

        // ---- chat template token ids ----
        const int sot = im.tok.piece_to_id("<start_of_turn>");
        const int eot = im.tok.piece_to_id("<end_of_turn>");
        std::vector<int> ids;
        ids.push_back(im.tok.bos_id());
        ids.push_back(sot);
        for (int t : im.tok.encode("user\n" + prompt)) ids.push_back(t);
        ids.push_back(eot);
        for (int t : im.tok.encode("\n")) ids.push_back(t);
        ids.push_back(sot);
        for (int t : im.tok.encode("model\n")) ids.push_back(t);

        im.kcache.assign(c.n_layers, mx::array(0.0f));
        im.vcache.assign(c.n_layers, mx::array(0.0f));

        const float scale = 1.0f / std::sqrt(static_cast<float>(c.query_pre_attn_scalar));
        const int rep = c.n_heads / c.n_kv;

        // forward a chunk of token ids at absolute positions [offset, offset+T) ->
        // logits for the LAST position. Updates the KV cache.
        auto forward_last = [&](const std::vector<int>& chunk, int offset) -> mx::array {
            const int T = static_cast<int>(chunk.size());
            std::vector<int> idv(chunk.begin(), chunk.end());
            mx::array tok_ids = mx::array(idv.data(), {T}, mx::int32);
            mx::array x = mx::multiply(mx::take(im.embed, tok_ids, 0), k(std::sqrt((float)c.hidden)));  // [T,hidden]

            // additive causal mask [T, offset+T] (0 allowed, -inf disallowed)
            mx::array mask = mx::array(0.0f);
            const int S = offset + T;
            if (T > 1) {
                std::vector<float> md(static_cast<size_t>(T) * S, 0.0f);
                for (int i = 0; i < T; ++i)
                    for (int j = 0; j < S; ++j)
                        if (j > offset + i) md[static_cast<size_t>(i) * S + j] = -1e9f;
                mask = mx::astype(mx::array(md.data(), {T, S}, mx::float32), kDt);
            }

            for (int l = 0; l < c.n_layers; ++l) {
                const std::string p = "model.layers." + std::to_string(l) + ".";
                auto attn = im.ql(p + "self_attn.q_proj");
                auto r = Impl::rmsnorm(x, in_ln[l], c.eps);
                auto q = im.ql(p + "self_attn.q_proj")(r);   // [T, n_heads*hd]
                auto kk = im.ql(p + "self_attn.k_proj")(r);  // [T, n_kv*hd]
                auto vv = im.ql(p + "self_attn.v_proj")(r);
                // [T, H, hd] -> [H, T, hd]
                q = mx::swapaxes(mx::reshape(q, {T, c.n_heads, c.head_dim}), 0, 1);
                kk = mx::swapaxes(mx::reshape(kk, {T, c.n_kv, c.head_dim}), 0, 1);
                vv = mx::swapaxes(mx::reshape(vv, {T, c.n_kv, c.head_dim}), 0, 1);
                q = mxf::rope(q, c.head_dim, /*traditional=*/false, std::optional<float>(c.rope_theta), 1.0f, offset);
                kk = mxf::rope(kk, c.head_dim, false, std::optional<float>(c.rope_theta), 1.0f, offset);
                // append to KV cache along the time axis (axis 1)
                if (offset == 0) { im.kcache[l] = kk; im.vcache[l] = vv; }
                else {
                    im.kcache[l] = mx::concatenate({im.kcache[l], kk}, 1);
                    im.vcache[l] = mx::concatenate({im.vcache[l], vv}, 1);
                }
                auto kc = mx::repeat(im.kcache[l], rep, 0);   // GQA: [n_heads, S, hd]
                auto vc = mx::repeat(im.vcache[l], rep, 0);
                auto scores = mx::multiply(mx::matmul(q, mx::swapaxes(kc, 1, 2)), k(scale));  // [H,T,S]
                scores = mx::multiply(mx::tanh(mx::divide(scores, k(c.attn_softcap))), k(c.attn_softcap));
                if (T > 1) scores = mx::add(scores, mask);
                auto probs = mx::softmax(scores, -1, /*precise=*/true);
                auto o = mx::matmul(probs, vc);                       // [H,T,hd]
                o = mx::reshape(mx::swapaxes(o, 0, 1), {T, c.n_heads * c.head_dim});
                o = im.ql(p + "self_attn.o_proj")(o);                 // [T, hidden]
                x = mx::add(x, Impl::rmsnorm(o, post_attn_ln[l], c.eps));

                auto r2 = Impl::rmsnorm(x, pre_ffn_ln[l], c.eps);
                auto gate = gelu_tanh(im.ql(p + "mlp.gate_proj")(r2));
                auto up = im.ql(p + "mlp.up_proj")(r2);
                auto m = im.ql(p + "mlp.down_proj")(mx::multiply(gate, up));
                x = mx::add(x, Impl::rmsnorm(m, post_ffn_ln[l], c.eps));
            }
            x = Impl::rmsnorm(x, final_ln, c.eps);
            // last position only
            auto last = mx::take(x, mx::array(T - 1, mx::int32), 0);   // [hidden]
            last = mx::reshape(last, {1, c.hidden});
            auto logits = mx::quantized_matmul(last, im.w.at("model.embed_tokens.weight"),
                                               im.w.at("model.embed_tokens.scales"),
                                               im.w.at("model.embed_tokens.biases"), true, c.group_size, c.bits);
            logits = mx::multiply(mx::tanh(mx::divide(logits, k(c.final_softcap))), k(c.final_softcap));
            return logits;  // [1, vocab]
        };

        using clk = std::chrono::steady_clock;
        std::vector<int> out_ids;
        const auto t_pf = clk::now();
        auto logits = forward_last(ids, 0);
        mx::eval(logits);  // force the prefill compute so we time it accurately
        const double prefill_s = std::chrono::duration<double>(clk::now() - t_pf).count();
        const int vocab = logits.shape().back();
        int pos = static_cast<int>(ids.size());
        int blank_run = 0;             // consecutive whitespace-only pieces
        int repeat_run = 0, last_tok = -1;  // consecutive identical tokens
        // Greedy repetition penalty over GENERATED tokens only (NOT the prompt —
        // we want the model to copy names out of the retrieved context). HF-style:
        // a seen token's logit /= penalty if positive, else *= penalty. This is
        // what stops the ",,,," / whitespace degeneration greedy falls into and
        // lets the model actually produce the list. The fuller sampler
        // (temperature/top-p/top-k/seed) is tracked in #104.
        constexpr float kRepPenalty = 1.3f;
        std::vector<float> pen(static_cast<size_t>(vocab), 1.0f);
        const auto t_dec = clk::now();
        for (int step = 0; step < max_new_tokens; ++step) {
            mx::array penalty = mx::astype(mx::array(pen.data(), {1, vocab}, mx::float32), kDt);
            mx::array adj = mx::where(mx::greater(logits, k(0.0f)),
                                      mx::divide(logits, penalty),
                                      mx::multiply(logits, penalty));
            mx::array next = mx::argmax(adj, /*axis=*/-1, /*keepdims=*/false);
            mx::eval(next);
            const int tok = static_cast<int>(next.item<uint32_t>());
            if (std::getenv("GEMMA_DEBUG")) {
                auto mxl = mx::max(logits); auto mnl = mx::min(logits); mx::eval({mxl, mnl});
                std::fprintf(stderr, "[dbg] step=%d pos=%d tok=%d max=%.4f min=%.4f piece='%s'\n",
                             step, pos, tok, (float)mxl.item<float>(), (float)mnl.item<float>(),
                             im.tok.decode({tok}).c_str());
            }
            if (tok == im.tok.eos_id() || tok == eot) break;
            // backstops for any residual loop the penalty doesn't break
            repeat_run = (tok == last_tok) ? repeat_run + 1 : 0;
            last_tok = tok;
            if (repeat_run >= 8) break;
            const std::string piece = im.tok.decode({tok});
            const bool blank =
                !piece.empty() && piece.find_first_not_of(" \t\r\n") == std::string::npos;
            if (blank && ++blank_run >= 12) break;
            if (!blank) blank_run = 0;
            if (tok >= 0 && tok < vocab) pen[static_cast<size_t>(tok)] = kRepPenalty;  // mark seen
            out_ids.push_back(tok);
            if (on_piece) on_piece(piece);
            logits = forward_last({tok}, pos);
            ++pos;
        }
        const double decode_s = std::chrono::duration<double>(clk::now() - t_dec).count();
        if (stats) {
            stats->prompt_tokens = static_cast<int>(ids.size());
            stats->gen_tokens = static_cast<int>(out_ids.size());
            stats->prefill_s = prefill_s;
            stats->decode_s = decode_s;
        }
        if (std::getenv("GEMMA_STATS")) {
            const double dsafe = decode_s > 0 ? decode_s : 1e-9;
            std::fprintf(stderr,
                "[stats] prefill: %zu tok in %.3f s = %.1f tok/s | decode: %zu tok in %.3f s = %.1f tok/s | peak GPU mem %.0f MB\n",
                ids.size(), prefill_s, static_cast<double>(ids.size()) / (prefill_s > 0 ? prefill_s : 1e-9),
                out_ids.size(), decode_s, static_cast<double>(out_ids.size()) / dsafe,
                static_cast<double>(mx::get_peak_memory()) / 1e6);
        }
        std::string text = im.tok.decode(out_ids);
        const auto last_nonws = text.find_last_not_of(" \t\r\n");
        text.erase(last_nonws == std::string::npos ? 0 : last_nonws + 1);
        return text;
    } catch (const std::exception& e) {
        return fail(std::string("gemma generate: ") + e.what());
    }
}

}  // namespace model

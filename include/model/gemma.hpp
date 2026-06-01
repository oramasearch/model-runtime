#pragma once
// model::Gemma — a real Gemma-2 language model in pure C++ on mlx::core (D26,
// no Python). Loads mlx-community 4-bit safetensors + the SentencePiece
// tokenizer, and generates text (greedy, KV-cache). Built with ORAMA_BACKEND_MLX.
// Architecture constants below are gemma-2-2b-it (Gemma2ForCausalLM).
#include <functional>
#include <memory>
#include <string>

#include "model/error.hpp"

namespace model {

struct GemmaConfig {
    int n_layers = 26;
    int n_heads = 8;
    int n_kv = 4;            // grouped-query attention
    int head_dim = 256;      // note: n_heads*head_dim (2048) != hidden (2304)
    int hidden = 2304;
    int inter = 9216;
    int vocab = 256000;
    float rope_theta = 10000.0f;
    float eps = 1e-6f;
    float attn_softcap = 50.0f;
    float final_softcap = 30.0f;
    int query_pre_attn_scalar = 256;  // attn scale = 1/sqrt(this)
    int group_size = 64;
    int bits = 4;
};

// Optional timing returned from generate() (for benchmarking).
struct GenStats {
    int prompt_tokens = 0;
    int gen_tokens = 0;
    double prefill_s = 0.0;
    double decode_s = 0.0;
};

class Gemma {
public:
    Gemma();
    ~Gemma();
    Gemma(Gemma&&) noexcept;
    Gemma& operator=(Gemma&&) noexcept;
    Gemma(const Gemma&) = delete;
    Gemma& operator=(const Gemma&) = delete;

    // `dir` holds model.safetensors + tokenizer.model (config is baked above).
    [[nodiscard]] static Result<Gemma> load(const std::string& dir);

    // Greedy generation. Wraps `prompt` in the Gemma chat template, prefills,
    // then decodes up to max_new_tokens (stops at <eos>/<end_of_turn>),
    // streaming each decoded piece via on_piece. Returns the full text.
    [[nodiscard]] Result<std::string> generate(
        const std::string& prompt, int max_new_tokens,
        const std::function<void(const std::string&)>& on_piece = {},
        GenStats* stats = nullptr);

    [[nodiscard]] const GemmaConfig& config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace model

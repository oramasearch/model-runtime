#pragma once
// model::Tokenizer — a thin C++ wrapper over SentencePiece (Gemma's tokenizer).
// Pure C++, no Python (D26). pimpl so this header carries no SentencePiece
// dependency. Part of the real-model path (#102); built with ORAMA_BACKEND_MLX.
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "model/error.hpp"

namespace model {

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // Load a SentencePiece model file (e.g. Gemma's tokenizer.model).
    [[nodiscard]] Result<void> load(const std::string& path);
    [[nodiscard]] bool loaded() const;

    [[nodiscard]] std::vector<int> encode(std::string_view text) const;
    [[nodiscard]] std::string decode(const std::vector<int>& ids) const;

    [[nodiscard]] int vocab_size() const;
    [[nodiscard]] int bos_id() const;
    [[nodiscard]] int eos_id() const;
    // Map a single piece (e.g. "<start_of_turn>") to its id, or -1. For building
    // chat templates where special tokens must be their own ids, not re-tokenized.
    [[nodiscard]] int piece_to_id(const std::string& piece) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace model

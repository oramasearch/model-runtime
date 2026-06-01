// Built only with ORAMA_BACKEND_MLX. Exercises the SentencePiece wrapper's
// lifecycle without a real model file (#102, D26). With a Gemma tokenizer.model
// present, ORAMA_TOKENIZER_MODEL can point at it for an encode/decode round-trip.
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <string>

#include "model/tokenizer.hpp"

TEST_CASE("Tokenizer is empty until loaded and fails cleanly on a bad path", "[tokenizer]") {
    model::Tokenizer tok;
    REQUIRE_FALSE(tok.loaded());
    auto r = tok.load("/nonexistent/tokenizer.model");
    REQUIRE_FALSE(r.has_value());
    REQUIRE_FALSE(tok.loaded());
    REQUIRE(tok.encode("hello world").empty());  // no model -> no tokens
    REQUIRE(tok.decode({1, 2, 3}).empty());
}

TEST_CASE("Tokenizer encode/decode round-trips a real model if provided", "[tokenizer]") {
    const char* model_path = std::getenv("ORAMA_TOKENIZER_MODEL");
    if (!model_path) { SUCCEED("set ORAMA_TOKENIZER_MODEL to a SentencePiece .model to run this"); return; }
    model::Tokenizer tok;
    REQUIRE(tok.load(model_path).has_value());
    REQUIRE(tok.loaded());
    REQUIRE(tok.vocab_size() > 0);
    const auto ids = tok.encode("hello world");
    REQUIRE_FALSE(ids.empty());
    REQUIRE_FALSE(tok.decode(ids).empty());
}

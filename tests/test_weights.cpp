// Built only with ORAMA_BACKEND_MLX. Round-trips a safetensors file with no
// Gemma weights needed — proves the pure-C++ weight loader (#102, D26).
#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <string>
#include <unordered_map>

#include <mlx/mlx.h>

#include "model/weights.hpp"

namespace mx = mlx::core;

TEST_CASE("Weights round-trips a safetensors file", "[weights]") {
    const std::string path = "/tmp/orama_weights_roundtrip.safetensors";
    std::unordered_map<std::string, mx::array> t;
    t.emplace("layer.w", mx::random::normal({4, 8}));
    t.emplace("layer.b", mx::zeros({8}, mx::float32));
    mx::save_safetensors(path, t);

    auto w = model::Weights::load(path);
    REQUIRE(w.has_value());
    REQUIRE(w->size() == 2);
    REQUIRE(w->has("layer.w"));
    REQUIRE(w->has("layer.b"));
    REQUIRE(w->at("layer.w").shape(0) == 4);
    REQUIRE(w->at("layer.w").shape(1) == 8);
    REQUIRE(w->at("layer.b").size() == 8);
    std::remove(path.c_str());
}

TEST_CASE("Weights load of a missing file fails cleanly (no throw)", "[weights]") {
    auto w = model::Weights::load("/nonexistent/nope.safetensors");
    REQUIRE_FALSE(w.has_value());
    REQUIRE_FALSE(w.error().message.empty());
}

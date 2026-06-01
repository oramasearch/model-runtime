// Real-MLX backend tests. Built only when ORAMA_BACKEND_MLX=ON (needs MLX).
#include <catch2/catch_test_macros.hpp>

#include <string>

#include "model/engine.hpp"

TEST_CASE("MLX backend is real (not the stub)", "[mlx]") {
    auto e = model::make_engine(model::Backend::MLX);
    REQUIRE(e != nullptr);
    REQUIRE(e->backend() == model::Backend::MLX);
    REQUIRE_FALSE(e->is_stub());
}

TEST_CASE("MLX LoRA train_step actually reduces the loss", "[mlx]") {
    auto e = model::make_engine(model::Backend::MLX);
    auto first = e->train_step({});
    REQUIRE(first.has_value());
    double last = first->loss;
    for (int i = 0; i < 40; ++i) {
        auto r = e->train_step({});
        REQUIRE(r.has_value());
        last = r->loss;
    }
    REQUIRE(last < first->loss);   // real autograd + AdamW descends the loss on Metal
}

TEST_CASE("MLX infer runs a real forward and still streams/steers (D25)", "[mlx]") {
    auto e = model::make_engine(model::Backend::MLX);
    int hits = 0;
    auto r = e->infer("anything", [&](std::string_view tok) -> std::string {
        ++hits;
        return tok == "[search]" ? std::string("(injected)") : std::string{};
    });
    REQUIRE(r.has_value());
    REQUIRE(r->tokens > 0);
    REQUIRE(hits == static_cast<int>(r->tokens));
    REQUIRE(r->injections == 1);
}

#include <catch2/catch_test_macros.hpp>

#include "model/backend.hpp"
#include "model/version.hpp"

TEST_CASE("version is set", "[scaffold]") {
    REQUIRE(model::kName == "orama-model-runtime");
    REQUIRE_FALSE(model::kVersion.empty());
}

TEST_CASE("backends are labeled", "[scaffold]") {
    REQUIRE(model::label(model::Backend::MLX) == "mlx");
    REQUIRE(model::label(model::Backend::LibTorch) == "libtorch");
    REQUIRE(model::label(model::Backend::LlamaCpp) == "llama.cpp");
    REQUIRE(model::label(model::Backend::CUDA) == "cuda");
}

TEST_CASE("default backend resolves for this platform", "[scaffold]") {
#if defined(__APPLE__)
    REQUIRE(model::default_backend() == model::Backend::MLX);
#else
    REQUIRE(model::default_backend() == model::Backend::LibTorch);
#endif
}

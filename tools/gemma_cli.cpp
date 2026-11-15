// SPDX-License-Identifier: MIT

// gemma-cli — load a real mlx-community Gemma-2 4-bit model (pure C++/MLX, no
// Python) and greedily generate. Built with ORAMA_BACKEND_MLX. Validation tool.
//   ./gemma-cli <model-dir> ["prompt"]
#include <cstdio>
#include <print>
#include <string>

#include "model/gemma.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::println(stderr, "usage: {} <model-dir> [prompt]", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const std::string prompt = argc > 2 ? argv[2] : "Explain what BM25 is in one sentence.";

    std::println("loading {} ...", dir);
    auto g = model::Gemma::load(dir);
    if (!g) { std::println(stderr, "load failed: {}", g.error().message); return 1; }

    std::print("prompt: {}\n--- generating ---\n", prompt);
    auto out = g->generate(prompt, 160, [](const std::string& piece) {
        std::print("{}", piece);
        std::fflush(stdout);
    });
    std::println("\n--- done ---");
    if (!out) { std::println(stderr, "generate failed: {}", out.error().message); return 1; }
    return 0;
}

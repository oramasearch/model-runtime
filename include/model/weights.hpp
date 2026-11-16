// SPDX-License-Identifier: MIT

#pragma once
// model::Weights — named tensors loaded from a .safetensors file into mlx
// arrays, in pure C++ (no Python, D26). The model forward reads its weights
// from here. Part of the real-model path (#102); built with ORAMA_BACKEND_MLX.
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/mlx.h>

#include "model/error.hpp"

namespace model {

class Weights {
public:
    // Load all tensors from a .safetensors file (e.g. an mlx-community Gemma-4
    // shard). 4-bit weights arrive as their packed tensor + sibling scale/bias
    // keys; dequantize via mx::dequantize at use sites.
    [[nodiscard]] static Result<Weights> load(const std::string& path);

    [[nodiscard]] bool has(const std::string& name) const { return t_.contains(name); }
    [[nodiscard]] const mlx::core::array& at(const std::string& name) const { return t_.at(name); }
    [[nodiscard]] std::size_t size() const { return t_.size(); }
    [[nodiscard]] std::vector<std::string> names() const;

private:
    std::unordered_map<std::string, mlx::core::array> t_;
};

}  // namespace model

#include "model/weights.hpp"

#include <exception>

namespace mx = mlx::core;

namespace model {

Result<Weights> Weights::load(const std::string& path) {
    try {
        auto loaded = mx::load_safetensors(path);  // {tensors, metadata}
        Weights w;
        w.t_ = std::move(loaded.first);
        return w;
    } catch (const std::exception& e) {
        // MLX throws on a bad/missing file; keep exceptions off our API (D7/D16).
        return fail(std::string("load_safetensors(") + path + "): " + e.what());
    }
}

std::vector<std::string> Weights::names() const {
    std::vector<std::string> out;
    out.reserve(t_.size());
    for (const auto& [k, _] : t_) out.push_back(k);
    return out;
}

}  // namespace model

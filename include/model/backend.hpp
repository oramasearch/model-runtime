#pragma once
// model-runtime — backend-agnostic train + infer behind ONE interface
// (Orama Platform D19). The same training/inference contract runs on a native
// backend per platform; NOT MLX-specific.
//
// Scaffold: the backend taxonomy + a default-pick helper. The Backend
// interface (train(), infer()) and the concrete backends land behind the
// CMake toggles with orama-platform#33/#34/#35/#36.
#include <string_view>

namespace model {

enum class Backend {
    MLX,        // Apple Metal / unified memory — macOS (first target)
    LibTorch,   // PyTorch C++ — Linux / Windows
    LlamaCpp,   // llama.cpp — Linux / Windows / CPU-friendly
    CUDA,       // NVIDIA — cloud GPUs
};

constexpr std::string_view label(Backend b) {
    switch (b) {
        case Backend::MLX:      return "mlx";
        case Backend::LibTorch: return "libtorch";
        case Backend::LlamaCpp: return "llama.cpp";
        case Backend::CUDA:     return "cuda";
    }
    return "unknown";
}

// The default native backend for the platform we're compiled on (D19).
constexpr Backend default_backend() {
#if defined(__APPLE__)
    return Backend::MLX;
#else
    return Backend::LibTorch;
#endif
}

}  // namespace model

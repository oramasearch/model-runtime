cmake_minimum_required(VERSION 3.28)
project(orama-model-runtime LANGUAGES CXX)

# Orama Platform — model-runtime: backend-agnostic model train + infer behind
# ONE interface (architecture D19). Backends are selected per platform:
#   MLX (macOS) · libtorch / llama.cpp (Linux/Windows) · CUDA (cloud).
# NOT MLX-specific. Small, single-responsibility module (D21).
# C++23 today; C++26 target (D7).
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Backend toggles — all OFF by default so the scaffold + CI build with no ML
# toolchain. Each flips on the matching backend lib when that backend lands.
option(ORAMA_BACKEND_MLX      "Build the Apple MLX backend (macOS)"        OFF)
option(ORAMA_BACKEND_LIBTORCH "Build the libtorch backend (Linux/Windows)" OFF)
option(ORAMA_BACKEND_LLAMACPP "Build the llama.cpp backend"                OFF)
option(ORAMA_BACKEND_CUDA     "Build the CUDA backend (cloud)"             OFF)

# Library first (D16). The interface is backend-free; backends are added behind
# the toggles with their impl issues (orama-platform#33/#34/#35/#36).
add_library(model-runtime INTERFACE)
add_library(orama::model-runtime ALIAS model-runtime)
target_include_directories(model-runtime INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

include(CTest)
if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1)
  FetchContent_MakeAvailable(Catch2)
  add_executable(model-runtime-tests tests/test_main.cpp)
  target_link_libraries(model-runtime-tests PRIVATE model-runtime Catch2::Catch2WithMain)
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(Catch)
  catch_discover_tests(model-runtime-tests)
endif()

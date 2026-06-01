#pragma once
// model::Result<T> = std::expected<T, model::Error>
// No exceptions cross a component's public API boundary (Orama Platform D7/D16).
#include <expected>
#include <string>
#include <utility>

namespace model {

struct Error {
    std::string message;
};

template <typename T>
using Result = std::expected<T, Error>;

inline std::unexpected<Error> fail(std::string msg) {
    return std::unexpected(Error{std::move(msg)});
}

}  // namespace model

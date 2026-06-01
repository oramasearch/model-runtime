#include <catch2/catch_test_macros.hpp>

#include <string>

#include "model/engine.hpp"

TEST_CASE("make_engine returns a stub for this platform", "[engine]") {
    auto e = model::make_engine();
    REQUIRE(e != nullptr);
    REQUIRE(e->is_stub());
    REQUIRE(e->backend() == model::default_backend());
    REQUIRE(e->name().find("stub") != std::string_view::npos);
}

TEST_CASE("infer streams tokens and folds in injected context (D25)", "[engine]") {
    auto e = model::make_engine();
    int seen = 0;
    auto r = e->infer("anything", [&](std::string_view tok) -> std::string {
        ++seen;
        return tok == "[search]" ? std::string("(top: Postgres)") : std::string{};
    });
    REQUIRE(r.has_value());
    REQUIRE(seen == static_cast<int>(r->tokens));   // callback fired once per token
    REQUIRE(r->injections == 1);                    // exactly the [search] token injected
    REQUIRE(r->text.find("(top: Postgres)") != std::string::npos);
}

TEST_CASE("infer without a callback still decodes", "[engine]") {
    auto e = model::make_engine();
    auto r = e->infer("anything");
    REQUIRE(r.has_value());
    REQUIRE(r->tokens > 0);
    REQUIRE(r->injections == 0);
}

TEST_CASE("train_step loss decreases monotonically", "[engine]") {
    auto e = model::make_engine();
    auto a = e->train_step({{"q", "a"}});
    auto b = e->train_step({{"q", "a"}});
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    REQUIRE(b->step == a->step + 1);
    REQUIRE(b->loss < a->loss);
}

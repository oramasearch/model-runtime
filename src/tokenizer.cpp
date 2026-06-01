#include "model/tokenizer.hpp"

#include <sentencepiece_processor.h>

namespace model {

struct Tokenizer::Impl {
    sentencepiece::SentencePieceProcessor sp;
    bool ok = false;
};

Tokenizer::Tokenizer() : impl_(std::make_unique<Impl>()) {}
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

Result<void> Tokenizer::load(const std::string& path) {
    const auto status = impl_->sp.Load(path);
    if (!status.ok()) return fail(std::string("tokenizer load: ") + status.ToString());
    impl_->ok = true;
    return {};
}

bool Tokenizer::loaded() const { return impl_->ok; }

std::vector<int> Tokenizer::encode(std::string_view text) const {
    std::vector<int> ids;
    if (impl_->ok) impl_->sp.Encode(std::string(text), &ids).IgnoreError();
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    if (impl_->ok) impl_->sp.Decode(ids, &out).IgnoreError();
    return out;
}

int Tokenizer::vocab_size() const { return impl_->ok ? impl_->sp.GetPieceSize() : 0; }
int Tokenizer::bos_id() const { return impl_->ok ? impl_->sp.bos_id() : -1; }
int Tokenizer::eos_id() const { return impl_->ok ? impl_->sp.eos_id() : -1; }

}  // namespace model

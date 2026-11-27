#include "qc/cli.hpp"

#include <algorithm>
#include <charconv>
#include <stdexcept>

namespace qc {

Args::Args(int argc, char** argv, std::vector<std::string> allowed_keys) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        const auto eq = a.find('=');
        if (a.substr(0, 2) != "--" || eq == std::string_view::npos || eq < 3) {
            throw std::invalid_argument("argument '" + std::string(a) + "' must have the form --key=value");
        }
        const std::string key(a.substr(2, eq - 2));
        if (std::find(allowed_keys.begin(), allowed_keys.end(), key) == allowed_keys.end()) {
            throw std::invalid_argument("unknown argument key '" + key + "'");
        }
        kv_[key] = std::string(a.substr(eq + 1));
    }
}

bool Args::has(std::string_view key) const { return kv_.find(key) != kv_.end(); }

template <class T>
static T parse_num(std::string_view key, const std::string& s) {
    T v{};
    const char* b = s.data();
    const char* e = b + s.size();
    const auto [p, ec] = std::from_chars(b, e, v);
    if (ec != std::errc{} || p != e) {
        throw std::invalid_argument("argument --" + std::string(key) + ": cannot parse '" + s + "'");
    }
    return v;
}

std::int64_t Args::get_int(std::string_view key, std::int64_t def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : parse_num<std::int64_t>(key, it->second);
}
double Args::get_double(std::string_view key, double def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : parse_num<double>(key, it->second);
}
std::uint64_t Args::get_u64(std::string_view key, std::uint64_t def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : parse_num<std::uint64_t>(key, it->second);
}
std::string Args::get_str(std::string_view key, std::string def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? std::move(def) : it->second;
}

}  // namespace qc

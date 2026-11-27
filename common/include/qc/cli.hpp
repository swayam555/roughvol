// Minimal --key=value command-line parser.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace qc {
class Args {
  public:
    // Parses argv[1..]; every argument must be --key=value with key in allowed_keys, else std::invalid_argument.
    Args(int argc, char** argv, std::vector<std::string> allowed_keys);
    [[nodiscard]] std::int64_t get_int(std::string_view key, std::int64_t def) const;
    [[nodiscard]] double get_double(std::string_view key, double def) const;
    [[nodiscard]] std::uint64_t get_u64(std::string_view key, std::uint64_t def) const;
    [[nodiscard]] std::string get_str(std::string_view key, std::string def) const;
    [[nodiscard]] bool has(std::string_view key) const;

  private:
    std::map<std::string, std::string, std::less<>> kv_;
};
}  // namespace qc

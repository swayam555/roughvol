// CSV result writer.
#pragma once
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <string>
#include <variant>
#include <vector>

namespace qc {
class CsvWriter {
  public:
    // Opens path (creating parent directories) and writes the header row. Throws std::runtime_error on failure.
    CsvWriter(const std::string& path, std::vector<std::string> header);
    // Writes one row; doubles are printed with 17 significant digits; strings are quoted if needed.
    // Throws std::invalid_argument if the column count differs from the header.
    void row(std::initializer_list<std::variant<double, std::int64_t, std::string>> cells);

  private:
    std::ofstream out_;
    std::size_t ncols_;
};
}  // namespace qc

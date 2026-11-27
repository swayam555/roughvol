#include "qc/csv.hpp"

#include <cstdio>
#include <filesystem>
#include <stdexcept>

namespace qc {

static std::string quote(const std::string& s) {
    if (s.find_first_of(",\"\n\r") == std::string::npos) return s;
    std::string r = "\"";
    for (char c : s) {
        if (c == '"') r += '"';
        r += c;
    }
    return r + '"';
}

CsvWriter::CsvWriter(const std::string& path, std::vector<std::string> header) : ncols_(header.size()) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    out_.open(path, std::ios::out | std::ios::trunc);
    if (!out_) throw std::runtime_error("CsvWriter: cannot open '" + path + "'");
    for (std::size_t i = 0; i < header.size(); ++i) out_ << (i ? "," : "") << quote(header[i]);
    out_ << '\n';
}

void CsvWriter::row(std::initializer_list<std::variant<double, std::int64_t, std::string>> cells) {
    if (cells.size() != ncols_) {
        throw std::invalid_argument("CsvWriter::row: " + std::to_string(cells.size()) + " cells, header has " +
                                    std::to_string(ncols_));
    }
    std::size_t i = 0;
    for (const auto& c : cells) {
        if (i++) out_ << ',';
        if (const auto* d = std::get_if<double>(&c)) {
            char buf[40];
            std::snprintf(buf, sizeof buf, "%.17g", *d);
            out_ << buf;
        } else if (const auto* n = std::get_if<std::int64_t>(&c)) {
            out_ << *n;
        } else {
            out_ << quote(std::get<std::string>(c));
        }
    }
    out_ << '\n';
    if (!out_) throw std::runtime_error("CsvWriter::row: write failed (disk full or closed stream?)");
}

}  // namespace qc

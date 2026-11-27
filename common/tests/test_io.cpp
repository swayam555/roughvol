#include <filesystem>
#include <fstream>
#include <qc/cli.hpp>
#include <qc/csv.hpp>
#include <qc/test.hpp>
#include <sstream>

TEST_CASE("Args parsing") {
    char a0[] = "prog", a1[] = "--n=42", a2[] = "--x=1.5", a3[] = "--name=abc", a4[] = "--seed=18446744073709551615";
    char* argv[] = {a0, a1, a2, a3, a4};
    qc::Args args(5, argv, {"n", "x", "name", "seed", "unused"});
    CHECK(args.get_int("n", 0) == 42);
    CHECK(args.get_double("x", 0) == 1.5);
    CHECK(args.get_str("name", "") == "abc");
    CHECK(args.get_u64("seed", 0) == 18446744073709551615ull);
    CHECK(args.get_int("unused", 7) == 7);
    CHECK(args.has("n"));
    CHECK_FALSE(args.has("unused"));
    char b1[] = "--bad=1";
    char* argv2[] = {a0, b1};
    CHECK_THROWS_AS(qc::Args(2, argv2, {"n"}), std::invalid_argument);
    char c1[] = "--n=abc";
    char* argv3[] = {a0, c1};
    qc::Args bad(2, argv3, {"n"});
    CHECK_THROWS_AS((void)bad.get_int("n", 0), std::invalid_argument);
}

TEST_CASE("CsvWriter") {
    const auto dir = std::filesystem::temp_directory_path() / "qc_csv_test_dir" / "sub";
    const auto path = (dir / "o.csv").string();
    {
        qc::CsvWriter w(path, {"a", "b", "c"});
        w.row({0.1, std::int64_t{7}, std::string("x,y")});
        CHECK_THROWS_AS(w.row({1.0}), std::invalid_argument);
    }
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    CHECK(ss.str() == "a,b,c\n0.10000000000000001,7,\"x,y\"\n");
    std::filesystem::remove_all(dir.parent_path());
}

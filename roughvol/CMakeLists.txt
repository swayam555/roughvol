# roughvol: rough Bergomi hybrid-scheme Monte Carlo (namespace rv). Sources are listed explicitly.
add_library(rv STATIC
  src/version.cpp)
target_include_directories(rv PUBLIC include)
target_link_libraries(rv PUBLIC qc)
qc_target_defaults(rv)

qc_add_test(rv.test_dual tests/test_dual.cpp "unit")

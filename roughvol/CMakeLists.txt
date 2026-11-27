# roughvol: rough Bergomi hybrid-scheme Monte Carlo (namespace rv). Sources are listed explicitly.
add_library(rv STATIC
  src/version.cpp src/black_scholes.cpp src/rbergomi.cpp src/pricer.cpp)
target_include_directories(rv PUBLIC include)
target_link_libraries(rv PUBLIC qc)
qc_target_defaults(rv)

qc_add_test(rv.test_dual tests/test_dual.cpp "unit")
qc_add_test(rv.test_bs_iv tests/test_bs_iv.cpp "unit")
qc_add_test(rv.test_kernel tests/test_kernel.cpp "unit")
qc_add_test(rv.test_paths  tests/test_paths.cpp  "unit;stat")
qc_add_test(rv.test_pricer tests/test_pricer.cpp "stat")
qc_add_test(rv.test_determinism tests/test_determinism.cpp "unit;parallel")

add_executable(rv_price apps/rv_price.cpp)
target_link_libraries(rv_price PRIVATE rv)
qc_target_defaults(rv_price)
qc_add_test(rv.test_greeks tests/test_greeks.cpp "stat")

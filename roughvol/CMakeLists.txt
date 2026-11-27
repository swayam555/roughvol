# roughvol: rough Bergomi hybrid-scheme Monte Carlo (namespace rv). Sources are listed explicitly.
add_library(rv STATIC
  src/version.cpp src/black_scholes.cpp src/rbergomi.cpp src/pricer.cpp src/lm.cpp src/calibration.cpp)
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
qc_add_test(rv.test_lm tests/test_lm.cpp "unit")
qc_add_test(rv.test_calibration tests/test_calibration.cpp "stat")
qc_add_test(rv.test_calibration_slow tests/test_calibration_slow.cpp "slow")

add_executable(rv_calibrate apps/rv_calibrate.cpp)
target_link_libraries(rv_calibrate PRIVATE rv)
qc_target_defaults(rv_calibrate)

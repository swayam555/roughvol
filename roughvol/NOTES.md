# roughvol — implementation notes

## B1 Dual numbers
- `Dual<N>` + generic math in `dual.hpp`/`math.hpp`. `pow(Dual, Dual)` takes its value from `std::pow` (so the value is exactly the double function) and the derivative from d(exp(y ln x)).
- `rv` is a STATIC library from the start (`src/version.cpp` is a tiny permanent TU so the library exists before B2).
- test_dual: all pass (release, ubsan).

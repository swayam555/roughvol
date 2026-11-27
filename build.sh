#!/usr/bin/env bash
# ./build.sh [preset] [--test] [--clean] [--target T] [-j N] [-- <extra ctest args>]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
VENV=$ROOT/.toolchain/.venv
if [[ ! -x $VENV/bin/python ]]; then
  echo "error: $VENV/bin/python missing (toolchain venv not found)" >&2; exit 1
fi
mkdir -p "$ROOT/.toolchain/bin"
for pair in zig-cc:cc zig-c++:c++ zig-ar:ar zig-ranlib:ranlib; do
  f=$ROOT/.toolchain/bin/${pair%%:*}
  if [[ ! -x $f ]]; then
    printf '#!/bin/sh\nexec %s/bin/python -m ziglang %s "$@"\n' "$VENV" "${pair##*:}" > "$f"
    chmod +x "$f"
  fi
done
export PATH=$VENV/bin:$PATH

PRESET=release; TEST=0; CLEAN=0; TARGET=""; N=""; EXTRA=()
while [[ $# -gt 0 ]]; do
  case $1 in
    release|relwithdebinfo|debug|ubsan|tsan) PRESET=$1 ;;
    --test) TEST=1 ;;
    --clean) CLEAN=1 ;;
    --target) TARGET=$2; shift ;;
    -j) N=$2; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "error: unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done

BTYPE=Release; SAN=none; FLAGS=""; FILTER=(-LE bench)
case $PRESET in
  release) BTYPE=Release; FLAGS="-O3 -DNDEBUG"; FILTER=(-LE bench) ;;
  relwithdebinfo) BTYPE=RelWithDebInfo; FLAGS="-O2 -g"; FILTER=(-LE "slow|bench") ;;
  debug) BTYPE=Debug; FLAGS="-O0 -g"; FILTER=(-LE "slow|bench") ;;
  ubsan) BTYPE=Debug; SAN=undefined; FLAGS="-O1 -g"; FILTER=(-LE "slow|bench") ;;
  tsan) BTYPE=Debug; SAN=thread; FLAGS="-O1 -g"; FILTER=(-L parallel) ;;
esac

BDIR=$ROOT/build/$PRESET
[[ $CLEAN -eq 1 ]] && rm -rf "$BDIR"
if [[ ! -f $BDIR/build.ninja ]]; then
  cmake -S "$ROOT" -B "$BDIR" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ROOT/cmake/zig-toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=$BTYPE -DCMAKE_CXX_FLAGS_${BTYPE^^}="$FLAGS" -DCMAKE_C_FLAGS_${BTYPE^^}="$FLAGS" \
    -DQC_SANITIZER=$SAN
fi
cmake --build "$BDIR" -j "${N:-${QC_JOBS:-8}}" ${TARGET:+--target $TARGET}
if [[ $TEST -eq 1 ]]; then
  cd "$BDIR"
  ctest --output-on-failure -j 4 "${FILTER[@]}" "${EXTRA[@]}"
fi

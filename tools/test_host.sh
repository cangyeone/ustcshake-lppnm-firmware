#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
task_tmp=$(mktemp -d)
trap 'rm -rf "$task_tmp"' EXIT INT TERM
cxx=${CXX:-c++}
"$cxx" -std=c++17 -O2 -Wall -Wextra -Werror -I include tests/core_test.cpp -o "$task_tmp/core_test"
"$cxx" -std=c++17 -O2 -Wall -Wextra -Werror -I include tests/paper_test.cpp -o "$task_tmp/paper_test"
"$cxx" -std=c++17 -O2 -Wall -Wextra -Werror model/test_native.cpp -o "$task_tmp/model_test"
"$task_tmp/core_test"
"$task_tmp/paper_test"
"$task_tmp/model_test"
echo 'All firmware and incremental-runtime host tests passed.'

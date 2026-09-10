#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_TOOLS_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SCRIPT_DIR}"

clang++ --target=wasm32 -fno-builtin -I "${RL_TOOLS_DIR}/include" -DWASM -c l2f.cpp -o l2f.o
if which wasm-ld > /dev/null 2>/dev/null; then
  wasm-ld --no-entry\
  --export=initial_parameters\
  --export=init\
  --export=sample_initial_parameters\
  --export=sample_initial_state\
  --export=set_action\
  --export=observe\
  --export=step\
  --export=state_size\
  --export=action_dim\
  --export=observation_dim\
  l2f.o -o l2f.wasm
else
  echo "wasm-ld not found, install with sudo apt install lld"
  exit 1
fi
# sudo apt install wabt
if which wasm-objdump > /dev/null 2>/dev/null; then
  wasm-objdump -hx l2f.wasm
fi

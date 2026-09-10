set -e

clang++ --target=wasm32 -fno-builtin -I ../../include -DWASM -c l2f.cpp -o l2f.o
if which wasm-ld > /dev/null 2>/dev/null; then
  wasm-ld --no-entry\
  --export=init\
  --export=init\
  --export=sample_initial_parameters\
  --export=sample_initial_state\
  --export=set_action\
  --export=get_observation\
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
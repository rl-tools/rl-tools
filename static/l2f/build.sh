clang++ --target=wasm32 -c l2f.cpp -o l2f.o
if which wasm-ld > /dev/null 2>/dev/null; then
  wasm-ld --no-entry --export=hello l2f.o -o l2f.wasm
else
  echo "wasm-ld not found, install with sudo apt install lld"
  exit 1
fi
# sudo apt install wabt
if which wasm-objdump > /dev/null 2>/dev/null; then
  wasm-objdump -hx l2f.wasm
fi
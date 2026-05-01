#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <h5_checkpoint> <openmv_mount>" >&2
    exit 1
fi

CKPT="$1"
MOUNT="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "no active virtualenv (VIRTUAL_ENV unset); activate one before running" >&2
    exit 1
fi
PY="$VIRTUAL_ENV/bin/python3"
VELA="${VELA:-$VIRTUAL_ENV/bin/vela}"
BIN_LIMIT="${BIN_LIMIT:-13}"

if [ ! -x "$PY" ]; then
    echo "missing python in active venv: $PY" >&2
    exit 1
fi
if [ ! -f "$CKPT" ]; then
    echo "missing checkpoint: $CKPT" >&2
    exit 1
fi
if [ ! -d "$MOUNT" ]; then
    echo "openmv mount not found: $MOUNT" >&2
    exit 1
fi

CKPT_DIR="$(cd "$(dirname "$CKPT")" && pwd)"
CKPT_BASE="$CKPT_DIR/$(basename "${CKPT%.h5}")"
VELA_INI="${VELA_INI:-$HOME/.config/OpenMV/openmvide/firmware/OPENMV_AE3/vela.ini}"

echo "==> wiping existing checkpoint/data from $MOUNT"
find "$MOUNT" -maxdepth 1 -type f \
    \( -name "*.tflite" -o -name "*.bin" -o -name "*.example_meta.json" \) \
    -print -delete

echo "==> converting $CKPT (bin samples: first $BIN_LIMIT)"
"$PY" "$REPO_ROOT/tools/hdf5_to_tflite.py" \
    --quantize int8 \
    --no-split-image-input \
    --example-bin-limit "$BIN_LIMIT" \
    "$CKPT"

INT8_TFLITE="${CKPT_BASE}.int8.tflite"
if [ ! -f "$INT8_TFLITE" ]; then
    echo "int8 tflite not found: $INT8_TFLITE" >&2
    exit 1
fi
if [ ! -x "$VELA" ]; then
    echo "missing vela executable: $VELA" >&2
    echo "Set VELA=/path/to/vela or install ethos-u-vela in .venv." >&2
    exit 1
fi
if [ ! -f "$VELA_INI" ]; then
    echo "missing Vela ini: $VELA_INI" >&2
    echo "Set VELA_INI=/path/to/vela.ini if your OpenMV firmware path differs." >&2
    exit 1
fi

echo "==> running vela on $INT8_TFLITE"
"$VELA" \
    --accelerator-config ethos-u55-256 \
    --config "$VELA_INI" \
    --system-config RTSS_HP_SRAM_OSPI \
    --memory-mode Shared_Sram \
    --output-dir "$CKPT_DIR" \
    "$INT8_TFLITE"

VELA_OUT="${CKPT_BASE}.int8_vela.tflite"
if [ ! -f "$VELA_OUT" ]; then
    echo "vela output not found: $VELA_OUT" >&2
    exit 1
fi

echo "==> copying artifacts to $MOUNT"
cp -v "$SCRIPT_DIR/main.py" "$MOUNT/main.py"
cp -v "$VELA_OUT" "$MOUNT/"
cp -v "${CKPT_BASE}".example_input.*.bin "$MOUNT/"
cp -v "${CKPT_BASE}.example_output.bin" "$MOUNT/"
cp -v "${CKPT_BASE}.example_meta.json" "$MOUNT/"
for extra in example_int8_output.bin example_int8_output_raw.bin; do
    src="${CKPT_BASE}.${extra}"
    if [ -f "$src" ]; then
        cp -v "$src" "$MOUNT/"
    fi
done

sync
echo "==> done"

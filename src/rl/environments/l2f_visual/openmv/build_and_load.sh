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
PY="$REPO_ROOT/.venv/bin/python3"
VELA="$REPO_ROOT/.venv/bin/vela"

CKPT_DIR="$(cd "$(dirname "$CKPT")" && pwd)"
CKPT_2="$CKPT_DIR/checkpoint_2examples.h5"
CKPT_512="$CKPT_DIR/checkpoint_512examples.h5"

for f in "$CKPT_2" "$CKPT_512"; do
    if [ ! -f "$f" ]; then
        echo "missing checkpoint: $f" >&2
        exit 1
    fi
done

if [ ! -d "$MOUNT" ]; then
    echo "openmv mount not found: $MOUNT" >&2
    exit 1
fi

echo "==> wiping existing checkpoint/data from $MOUNT"
find "$MOUNT" -maxdepth 1 -type f \
    \( -name "*.tflite" -o -name "*.bin" -o -name "*.example_meta.json" \) \
    -print -delete

CONVERT_ARGS=(--quantize int8 --split-image-input 6 --split-image-channels-per 3)

echo "==> converting $CKPT_2"
"$PY" "$REPO_ROOT/tools/hdf5_to_tflite.py" "${CONVERT_ARGS[@]}" "$CKPT_2"

echo "==> converting $CKPT_512"
"$PY" "$REPO_ROOT/tools/hdf5_to_tflite.py" "${CONVERT_ARGS[@]}" "$CKPT_512"

INT8_512="$CKPT_DIR/checkpoint_512examples.int8.tflite"
VELA_INI="$HOME/.config/OpenMV/openmvide/firmware/OPENMV_AE3/vela.ini"

echo "==> running vela on $INT8_512"
"$VELA" \
    --accelerator-config ethos-u55-256 \
    --config "$VELA_INI" \
    --system-config RTSS_HP_SRAM_OSPI \
    --memory-mode Shared_Sram \
    --output-dir "$CKPT_DIR" \
    "$INT8_512"

VELA_OUT="$CKPT_DIR/checkpoint_512examples.int8_vela.tflite"
if [ ! -f "$VELA_OUT" ]; then
    echo "vela output not found: $VELA_OUT" >&2
    exit 1
fi

echo "==> copying artifacts to $MOUNT"
cp -v "$VELA_OUT" "$MOUNT/"
cp -v "$CKPT_DIR"/checkpoint_2examples.example_input.*.bin "$MOUNT/"
cp -v "$CKPT_DIR/checkpoint_2examples.example_output.bin" "$MOUNT/"
cp -v "$CKPT_DIR/checkpoint_2examples.example_meta.json" "$MOUNT/"
for extra in example_int8_output.bin example_int8_output_raw.bin; do
    src="$CKPT_DIR/checkpoint_2examples.$extra"
    if [ -f "$src" ]; then
        cp -v "$src" "$MOUNT/"
    fi
done

sync
echo "==> done"

# OpenMV deployment

Single-checkpoint flow. The large HDF5 (e.g. `checkpoint_512examples.h5`) is
used for both int8 calibration and for the deployed companion bins; the number
of samples written to the bins is capped by `--example-bin-limit` to fit the
OpenMV flash budget.

## End-to-end (script)

```sh
src/rl/environments/l2f_visual/openmv/build_and_load.sh \
    <path>/checkpoint_<N>examples.h5 \
    /run/media/<you>/OPENMV
```

The script runs `hdf5_to_tflite.py --quantize int8 --split-image-input 6 --split-image-channels-per 3 --example-bin-limit 16`,
invokes Vela on the produced `.int8.tflite`, wipes stale artifacts from the
mount and copies the Vela-compiled tflite + `.example_input.*.bin` +
`.example_output.bin` + `.example_meta.json` + `.example_int8_output{,_raw}.bin`
across. The bin limit (16) is the number of samples `inference.py` will check
on boot.

## Manual steps

```sh
python3 tools/hdf5_to_tflite.py \
    --quantize int8 \
    --split-image-input 6 --split-image-channels-per 3 \
    --example-bin-limit 16 \
    <path>/checkpoint_<N>examples.h5

python3 -m ethosu.vela \
    --optimise Performance \
    --system-config RTSS_HP_SRAM_OSPI \
    --accelerator-config ethos-u55-256 \
    --memory-mode Shared_Sram \
    --config $HOME/.config/OpenMV/openmvide/firmware/OPENMV_AE3/vela.ini \
    --verbose-performance --verbose-cycle-estimate \
    --output-dir <path>/ \
    <path>/checkpoint_<N>examples.int8.tflite
```

Copy the Vela tflite + bin/json artifacts to the OpenMV mount, then open
`inference.py` in the OpenMV IDE and run.

```sh
mpremote connect /dev/serial/by-id/usb-OpenMV_OpenMV_Camera_085ce70000000000-if00 \
    run src/rl/environments/l2f_visual/openmv/inference.py
```

`inference.py` auto-detects the number of samples present in the bins and
iterates the float-reference + int8 strict-wiring check across all of them
before starting the live 100Hz policy loop.

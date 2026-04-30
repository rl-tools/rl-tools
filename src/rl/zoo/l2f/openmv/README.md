# OpenMV Attitude-Setpoint SAC Deployment

This mode runs the `rl_zoo_l2f_attitude_setpoint_sac` actor on OpenMV,
hosts a Wi-Fi AP, receives attitude setpoints over UDP, and writes motor
commands through the Crazyflie UART bridge.

## Build And Load

```sh
src/rl/zoo/l2f/openmv/build_and_load.sh \
    experiments/.../checkpoint.h5 \
    /run/media/$USER/OPENMV
```

The script runs:

```sh
.venv/bin/python3 tools/hdf5_to_tflite.py \
    --quantize int8 \
    --no-split-image-input \
    --example-bin-limit "${BIN_LIMIT:-13}" \
    <checkpoint.h5>
```

Then it runs Vela and copies `main.py`, the Vela TFLite, and the companion
`.bin`/`.json` self-check files to the OpenMV mount. Override paths with:

```sh
BIN_LIMIT=13 VELA=/path/to/vela VELA_INI=/path/to/vela.ini \
    src/rl/zoo/l2f/openmv/build_and_load.sh <checkpoint.h5> <openmv_mount>
```

## UDP Packet

The board listens on `0.0.0.0:5005` after creating AP `l2f-setpoint`
with password `attitude123`.

Packet format:

```text
little-endian <4sIffff
magic:      b"ASP1"
sequence:   uint32
roll:       float32 radians
pitch:      float32 radians
yaw_rate:   float32 radians/second
thrust_g:   float32 body-z thrust command in g
```

Setpoints are clamped to the training envelope:
`roll/pitch = +/-30 deg`, `yaw_rate = +/-2 rad/s`, `thrust = 0.4..1.4 g`.
If no valid packet arrives for 250 ms, the board switches to neutral
roll/pitch/yaw and `0.4 g` thrust.

## Gamepad Sender

Install host dependencies in the repo venv:

```sh
.venv/bin/python3 -m pip install gamepad-mapper pygame
```

Connect to the OpenMV AP, then run:

```sh
.venv/bin/python3 src/rl/zoo/l2f/openmv/gamepad_attitude_setpoint_udp.py --remap
```

The sender defaults to `192.168.4.1:5005`, 50 Hz, and requires the mapped
`arm` button to be held before non-idle setpoints are sent.

## Bench Checks

Before flight, run with props off and verify:

1. OpenMV boot self-check passes against the companion bins.
2. UDP packet age stays below the 250 ms failsafe timeout.
3. Roll, pitch, yaw-rate, and thrust signs match the diagnostics.
4. UART bridge output reaches the Crazyflie and arm/disarm behavior is correct.

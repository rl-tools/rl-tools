# OpenMV Attitude-Setpoint PPO Deployment

This mode runs an `rl_zoo_l2f_attitude_setpoint_ppo` actor on the OpenMV AE3
and sends direct motor actions to the Crazyflie UART offboard bridge. It is
intended for initial gimbal testing with a fixed setpoint:

```text
roll      = 0 rad
pitch     = 0 rad
yaw_rate  = 0 rad/s
thrust_g  = 1
```

There is no Wi-Fi, UDP, gamepad input, or host telemetry path.

## Build And Load

```sh
src/rl/zoo/l2f/openmv/build_and_load.sh \
    experiments/.../checkpoint.h5 \
    /run/media/$USER/OPENMV
```

The script converts the checkpoint to fp32 TFLite:

```sh
.venv/bin/python3 tools/hdf5_to_tflite.py \
    --quantize none \
    --no-split-image-input \
    --example-bin-limit "${BIN_LIMIT:-13}" \
    <checkpoint.h5>
```

It then runs Vela on the fp32 `.tflite` and copies `main.py`, the Vela TFLite,
and the fp32 companion `.bin`/`.json` self-check files to the OpenMV mount.
Override paths with:

```sh
BIN_LIMIT=13 VELA=/path/to/vela VELA_INI=/path/to/vela.ini \
    src/rl/zoo/l2f/openmv/build_and_load.sh <checkpoint.h5> <openmv_mount>
```

## Runtime Observation

The expected actor observation is:

```text
AttitudeSetpoint.OrientationWorldZ.AngularVelocity.ActionHistory(8)
```

This is 42 fp32 values:

```text
4   fixed attitude/thrust setpoint
3   world z vector in body frame
3   body angular velocity
32  action history, newest first
```

The live loop runs inference at 500 Hz (`CONTROL_SUBSTEPS = 5` over the
100 Hz training control frequency). The action history slots remain at the
training cadence: each action-history entry is the average over the five fast
control ticks since the previous history update.

The action history is initialized to the training hover action:

```text
2 * 0.6864 - 1 = 0.3728
```

## Crazyflie UART Frame

The OpenMV sends one 13-byte frame at 500 Hz. It never sets the
self-activation flag; offboard activation must come from the Crazyflie side.

```text
byte 0:    0x80 | flags; currently 0x80 with flags = 0
bytes 1-12: 12 bytes of 7-bit-packed raw payload, each with MSB clear
```

The unpacked raw payload is 10 bytes:

```text
raw[0..7]: four big-endian uint16 motor PWM values
raw[8..9]: big-endian CRC16-CCITT
```

The CRC is calculated over exactly 9 bytes: the full start/flags byte followed
by `raw[0..7]`. This matches the Crazyflie `uart1_bridge` receiver.

## Gimbal Checks

Before free flight, verify on the gimbal:

1. OpenMV boot self-check passes against the fp32 companion bins.
2. Level attitude reports `world_z` close to `(0, 0, 1)`.
3. Manual roll and pitch motion produce the expected `world_z` signs.
4. `u1br.framesOk` increments on the Crazyflie.
5. Neutral actions and PWM values are plausible before enabling props.

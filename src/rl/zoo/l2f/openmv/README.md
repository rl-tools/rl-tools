# OpenMV Attitude-Setpoint PPO Deployment

This mode runs an `rl_zoo_l2f_attitude_setpoint_ppo` actor on the OpenMV AE3
and sends direct motor actions to the Crazyflie UART offboard bridge. The
Crazyflie forwards the current controller attitude setpoint to the OpenMV over
the same UART link. If no fresh setpoint frame is received, the OpenMV falls
back to:

```text
roll      = 0 rad
pitch     = 0 rad
yaw_rate  = 0 rad/s
thrust_g  = 1
```

There is no Wi-Fi, UDP, or host telemetry path.

## Build And Load

```sh
src/rl/zoo/l2f/openmv/build_and_load.sh \
    experiments/.../checkpoint.h5 \
    /run/media/$USER/OPENMV
```

The script converts the checkpoint to int8 TFLite by default:

```sh
.venv/bin/python3 tools/hdf5_to_tflite.py \
    --quantize int8 \
    --no-split-image-input \
    --example-bin-limit "${BIN_LIMIT:-13}" \
    <checkpoint.h5>
```

It then runs Vela on `<checkpoint>.int8.tflite` and copies `main.py`, the Vela
TFLite, and the companion `.bin`/`.json` self-check files to the OpenMV mount.
The OpenMV runtime pre-quantizes the live observation into the model's int8
input scale before inference. Override paths or force the old fp32 flow with:

```sh
BIN_LIMIT=13 QUANTIZE=none VELA=/path/to/vela VELA_INI=/path/to/vela.ini \
    src/rl/zoo/l2f/openmv/build_and_load.sh <checkpoint.h5> <openmv_mount>
```

## Runtime Observation

The expected actor observation is:

```text
AttitudeSetpoint.OrientationWorldZ.AngularVelocity.ActionHistory(8)
```

This is 42 fp32 values:

```text
4   attitude/thrust setpoint from CF, with neutral fallback
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

The OpenMV sends one motor-action frame at 500 Hz. It never sets the
self-activation flag; offboard activation must come from the Crazyflie side.
The firmware parameter `u1br.defaultCtl` controls whether the normal Crazyflie
controller output is allowed when the UART offboard bridge is not engaged. The
deployment default is `0`, which means controller setpoints can still be
forwarded to OpenMV, but the normal CF controller PWM output is zeroed unless
the offboard bridge is actively engaged. Set it to `1` to restore normal
controller output when offboard is inactive.

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

## Crazyflie Setpoint Frame

The Crazyflie forwards the decoded commander setpoint to OpenMV at 100 Hz. The
firmware converts the values to the policy-facing `AttitudeSetpoint` units:

```text
roll      rad
pitch     rad
yaw_rate  rad/s
thrust_g  g, using u1br.thrust1g as the raw-thrust value for 1 g
```

The values are constrained to the training command distribution before they are
sent:

```text
tilt cone <= 30 deg
yaw_rate  in [-2, 2] rad/s
thrust_g  in [0.4, 1.4]
```

The CF-to-OpenMV frame uses the same MSB-start/7-bit-payload convention:

```text
byte 0:     0x80 | 0x02
bytes 1-13: 13 bytes of 7-bit-packed raw payload, each with MSB clear
```

The unpacked raw payload is 11 bytes:

```text
raw[0]:     sequence uint8
raw[1..2]:  int16 roll_rad * 10000, big-endian
raw[3..4]:  int16 pitch_rad * 10000, big-endian
raw[5..6]:  int16 yaw_rate_rad_s * 10000, big-endian
raw[7..8]: uint16 thrust_g * 10000, big-endian
raw[9..10]: big-endian CRC16-CCITT
```

The CRC is calculated over exactly 10 bytes: the full start/type byte followed
by `raw[0..8]`. If no valid frame arrives for 300 ms, OpenMV uses the neutral
fallback setpoint.

## Gimbal Checks

Before free flight, verify on the gimbal:

1. OpenMV boot self-check passes against the int8 companion bins.
2. Level attitude reports `world_z` close to `(0, 0, 1)`.
3. Manual roll and pitch motion produce the expected `world_z` signs.
4. `u1br.framesOk` increments on the Crazyflie.
5. Moving the controller changes the OpenMV diagnostic `sp=` values in radians,
   rad/s, and g.
6. Neutral actions and PWM values are plausible before enabling props.

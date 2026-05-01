#!/usr/bin/env python3
"""Send L2F AttitudeSetpoint UDP packets from a gamepad.

Default binary packet:
    <4sIIffff = b"ASP2", sequence, armed, roll_rad, pitch_rad, yaw_rate_rad_s, thrust_g
"""

import argparse
import math
import socket
import struct
import sys
import time


AXES = ["Roll", "Pitch", "Throttle", "Yaw"]
BUTTONS = ["arm"]
PACKET = struct.Struct("<4sIIffff")
MAGIC = b"ASP2"


def axis_value(axes, name, invert, deadzone, expo):
    value = float(axes.get(name, 0.0))
    if invert:
        value = -value
    if abs(value) < deadzone:
        return 0.0
    if deadzone > 0:
        value = math.copysign((abs(value) - deadzone) / (1.0 - deadzone), value)
    if expo != 0:
        value = (1.0 - expo) * value + expo * value * value * value
    return max(-1.0, min(1.0, value))


def asymmetric_thrust(axis, hover, thrust_min, thrust_max):
    if axis >= 0.0:
        return hover + axis * (thrust_max - hover)
    return hover + axis * (hover - thrust_min)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Map a gamepad to rl_zoo_l2f_attitude_setpoint_sac UDP setpoints."
    )
    ap.add_argument("--host", default="192.168.4.1", help="OpenMV AP address")
    ap.add_argument("--port", type=int, default=5005, help="OpenMV UDP port")
    ap.add_argument("--rate", type=float, default=50.0, help="send rate in Hz")
    ap.add_argument("--joystick", type=int, default=0, help="pygame joystick index")
    ap.add_argument("--mapping-name", default="l2f_attitude_setpoint", help="gamepad-mapper profile name")
    ap.add_argument("--remap", action="store_true", help="force interactive axis/button remapping")
    ap.add_argument("--broadcast", action="store_true", help="enable UDP broadcast on the socket")
    ap.add_argument("--require-arm", action=argparse.BooleanOptionalAction, default=True,
                    help="when true, send neutral idle-thrust setpoints unless the arm button is held")
    ap.add_argument("--roll-scale-deg", type=float, default=30.0)
    ap.add_argument("--pitch-scale-deg", type=float, default=30.0)
    ap.add_argument("--yaw-rate-scale", type=float, default=2.0, help="rad/s at full stick")
    ap.add_argument("--thrust-min-g", type=float, default=0.4)
    ap.add_argument("--thrust-hover-g", type=float, default=1.0)
    ap.add_argument("--thrust-max-g", type=float, default=1.4)
    ap.add_argument("--idle-thrust-g", type=float, default=0.4)
    ap.add_argument("--deadzone", type=float, default=0.05)
    ap.add_argument("--expo", type=float, default=0.25, help="0=linear, 1=cubic")
    ap.add_argument("--invert-roll", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--invert-pitch", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--invert-yaw", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--invert-throttle", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--print-every", type=float, default=1.0, help="status print interval in seconds")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.rate <= 0:
        raise SystemExit("--rate must be positive")
    if not (0.0 <= args.deadzone < 1.0):
        raise SystemExit("--deadzone must be in [0, 1)")
    if not (0.0 <= args.expo <= 1.0):
        raise SystemExit("--expo must be in [0, 1]")
    if not (args.thrust_min_g <= args.thrust_hover_g <= args.thrust_max_g):
        raise SystemExit("--thrust-hover-g must lie between min and max")

    try:
        import pygame
        from gamepad_mapper import load_or_map, read_gamepad
    except ImportError as e:
        print(
            "Missing dependency. Install in the repo venv with:\n"
            "  .venv/bin/python3 -m pip install gamepad-mapper pygame",
            file=sys.stderr,
        )
        raise SystemExit(2) from e

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() <= args.joystick:
        raise SystemExit("No game controller found at index %d" % args.joystick)

    joystick = pygame.joystick.Joystick(args.joystick)
    joystick.init()
    mapping = load_or_map(joystick, AXES, BUTTONS, force=args.remap, name=args.mapping_name)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if args.broadcast:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    target = (args.host, args.port)

    roll_scale = math.radians(args.roll_scale_deg)
    pitch_scale = math.radians(args.pitch_scale_deg)
    period = 1.0 / args.rate
    seq = 0
    next_send = time.monotonic()
    next_print = next_send
    last = (0.0, 0.0, 0.0, args.idle_thrust_g, False)

    print("sending ASP2 UDP setpoints to %s:%d from %s" % (args.host, args.port, joystick.get_name()))
    try:
        while True:
            now = time.monotonic()
            pygame.event.pump()
            axes, buttons = read_gamepad(joystick, mapping)
            arm_button = bool(buttons.get("arm", 0))
            armed = arm_button or not args.require_arm

            roll_axis = axis_value(axes, "Roll", args.invert_roll, args.deadzone, args.expo)
            pitch_axis = axis_value(axes, "Pitch", args.invert_pitch, args.deadzone, args.expo)
            yaw_axis = axis_value(axes, "Yaw", args.invert_yaw, args.deadzone, args.expo)
            throttle_axis = axis_value(axes, "Throttle", args.invert_throttle, args.deadzone, args.expo)

            if args.require_arm and not armed:
                roll = 0.0
                pitch = 0.0
                yaw_rate = 0.0
                thrust = args.idle_thrust_g
            else:
                roll = roll_axis * roll_scale
                pitch = pitch_axis * pitch_scale
                yaw_rate = yaw_axis * args.yaw_rate_scale
                thrust = asymmetric_thrust(
                    throttle_axis,
                    args.thrust_hover_g,
                    args.thrust_min_g,
                    args.thrust_max_g,
                )

            if now >= next_send:
                payload = PACKET.pack(MAGIC, seq & 0xFFFFFFFF, 1 if armed else 0,
                                      roll, pitch, yaw_rate, thrust)
                sock.sendto(payload, target)
                seq += 1
                next_send += period
                if next_send < now - period:
                    next_send = now + period
                last = (roll, pitch, yaw_rate, thrust, armed)

            if now >= next_print:
                print(
                    "seq=%d armed=%d roll=%+.3f pitch=%+.3f yaw_rate=%+.3f thrust_g=%.3f"
                    % (seq, last[4], last[0], last[1], last[2], last[3])
                )
                next_print = now + args.print_every

            sleep_s = min(max(next_send - time.monotonic(), 0.0), 0.01)
            if sleep_s:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()

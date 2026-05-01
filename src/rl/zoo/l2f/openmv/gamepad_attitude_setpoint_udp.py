#!/usr/bin/env python3
"""Send L2F AttitudeSetpoint UDP packets from a gamepad.

Default binary packet:
    <4sIIffff = b"ASP2", sequence, armed, roll_rad, pitch_rad, yaw_rate_rad_s, thrust_g
"""

import argparse
import csv
from datetime import datetime
import math
import os
import socket
import struct
import sys
import time


AXES = ["Roll", "Pitch", "Throttle", "Yaw"]
BUTTONS = ["arm"]
PACKET = struct.Struct("<4sIIffff")
MAGIC = b"ASP2"
TELEMETRY_MAGIC = b"L2FT"
TELEMETRY_VERSION = 1
TELEMETRY_PACKET = struct.Struct(
    "<4sHH" + "I" * 4 + "i" + "I" * 3 + "i" * 11 + "f" * 30
)
TELEMETRY_FIELDS = [
    "telemetry_version",
    "flags",
    "telemetry_seq",
    "control_tick",
    "openmv_t0_us",
    "openmv_t_actions_us",
    "last_setpoint_seq",
    "setpoint_packet_count",
    "bad_packet_count",
    "packet_age_us",
    "dt_us",
    "elapsed_us",
    "deadline_delay_us",
    "udp_us",
    "imu_us",
    "mahony_us",
    "state_us",
    "predict_us",
    "tx_us",
    "rx_us",
    "actions_us",
    "ax_orig_mg",
    "ay_orig_mg",
    "az_orig_mg",
    "gx_orig_mdps",
    "gy_orig_mdps",
    "gz_orig_mdps",
    "ax_mps2",
    "ay_mps2",
    "az_mps2",
    "gx_rad_s",
    "gy_rad_s",
    "gz_rad_s",
    "world_z_x",
    "world_z_y",
    "world_z_z",
    "mahony_bias_x_rad_s",
    "mahony_bias_y_rad_s",
    "mahony_bias_z_rad_s",
    "setpoint_roll_rad",
    "setpoint_pitch_rad",
    "setpoint_yaw_rate_rad_s",
    "setpoint_thrust_g",
    "action_raw_0",
    "action_raw_1",
    "action_raw_2",
    "action_raw_3",
    "action_0",
    "action_1",
    "action_2",
    "action_3",
]
TELEMETRY_CSV_FIELDS = [
    "host_time_s",
    "host_monotonic_s",
    "src_ip",
    "src_port",
] + TELEMETRY_FIELDS + [
    "tx_enabled",
    "udp_armed",
    "stale",
    "telemetry_lost_since_prev",
    "telemetry_lost_total",
]
FAIL_CLOSED_PACKETS = 5
FAIL_CLOSED_INTERVAL_S = 0.02


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
                    help="when true, mark UDP packets unarmed unless the arm button is held")
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
    ap.add_argument("--telemetry-csv", default="auto",
                    help="CSV path for OpenMV L2FT telemetry; 'auto' creates logs/openmv_telemetry_<timestamp>.csv, 'off' disables")
    return ap.parse_args()


def default_telemetry_csv_path():
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join("logs", "openmv_telemetry_%s.csv" % stamp)


def decode_telemetry(data):
    if len(data) != TELEMETRY_PACKET.size:
        return None
    values = TELEMETRY_PACKET.unpack(data)
    if values[0] != TELEMETRY_MAGIC or values[1] != TELEMETRY_VERSION:
        return None
    row = dict(zip(TELEMETRY_FIELDS, values[1:]))
    flags = int(row["flags"])
    row["tx_enabled"] = 1 if flags & 0x01 else 0
    row["udp_armed"] = 1 if flags & 0x02 else 0
    row["stale"] = 1 if flags & 0x04 else 0
    return row


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
    joystick_instance_id = joystick.get_instance_id() if hasattr(joystick, "get_instance_id") else None
    mapping = load_or_map(joystick, AXES, BUTTONS, force=args.remap, name=args.mapping_name)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if args.broadcast:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setblocking(False)
    target = (args.host, args.port)

    telemetry_csv_path = None
    telemetry_csv_file = None
    telemetry_writer = None
    telemetry_csv_arg = args.telemetry_csv.lower()
    if telemetry_csv_arg not in ("off", "none", "false", "0"):
        telemetry_csv_path = default_telemetry_csv_path() if telemetry_csv_arg == "auto" else args.telemetry_csv
        telemetry_dir = os.path.dirname(telemetry_csv_path)
        if telemetry_dir:
            os.makedirs(telemetry_dir, exist_ok=True)
        telemetry_csv_file = open(telemetry_csv_path, "w", newline="")
        telemetry_writer = csv.DictWriter(telemetry_csv_file, fieldnames=TELEMETRY_CSV_FIELDS)
        telemetry_writer.writeheader()

    roll_scale = math.radians(args.roll_scale_deg)
    pitch_scale = math.radians(args.pitch_scale_deg)
    period = 1.0 / args.rate
    seq = 0
    next_send = time.monotonic()
    next_print = next_send
    last = (0.0, 0.0, 0.0, args.idle_thrust_g, False)
    telemetry_stats = {
        "count": 0,
        "lost": 0,
        "last_seq": None,
        "last_row": None,
    }

    def send_packet(armed, roll, pitch, yaw_rate, thrust):
        nonlocal seq
        payload = PACKET.pack(MAGIC, seq & 0xFFFFFFFF, 1 if armed else 0,
                              roll, pitch, yaw_rate, thrust)
        sock.sendto(payload, target)
        seq += 1

    def poll_telemetry():
        while True:
            try:
                data, addr = sock.recvfrom(2048)
            except BlockingIOError:
                break
            row = decode_telemetry(data)
            if row is None:
                continue
            seq_rx = int(row["telemetry_seq"])
            lost_since_prev = 0
            last_seq = telemetry_stats["last_seq"]
            if last_seq is not None:
                delta = (seq_rx - last_seq) & 0xFFFFFFFF
                if 0 < delta < 0x80000000:
                    lost_since_prev = delta - 1
                    telemetry_stats["lost"] += lost_since_prev
            telemetry_stats["last_seq"] = seq_rx
            telemetry_stats["count"] += 1
            row["host_time_s"] = time.time()
            row["host_monotonic_s"] = time.monotonic()
            row["src_ip"] = addr[0]
            row["src_port"] = addr[1]
            row["telemetry_lost_since_prev"] = lost_since_prev
            row["telemetry_lost_total"] = telemetry_stats["lost"]
            telemetry_stats["last_row"] = row
            if telemetry_writer is not None:
                telemetry_writer.writerow(row)
                if telemetry_stats["count"] % 100 == 0:
                    telemetry_csv_file.flush()

    def send_disarm_burst(reason):
        print("%s; sending unarmed packets and exiting" % reason, file=sys.stderr)
        for _ in range(FAIL_CLOSED_PACKETS):
            try:
                send_packet(False, 0.0, 0.0, 0.0, args.idle_thrust_g)
            except OSError:
                break
            time.sleep(FAIL_CLOSED_INTERVAL_S)

    def fail_closed(reason):
        send_disarm_burst(reason)
        raise SystemExit(1)

    def check_joystick_events():
        for event in pygame.event.get((pygame.QUIT, pygame.JOYDEVICEREMOVED)):
            if event.type == pygame.QUIT:
                fail_closed("pygame quit event")
            event_instance_id = getattr(event, "instance_id", None)
            if (joystick_instance_id is None or event_instance_id is None or
                    event_instance_id == joystick_instance_id):
                fail_closed("gamepad removed")

    def check_joystick_attached():
        if hasattr(joystick, "get_attached") and not joystick.get_attached():
            fail_closed("gamepad no longer attached")

    print("sending ASP2 UDP setpoints to %s:%d from %s" % (args.host, args.port, joystick.get_name()))
    if telemetry_csv_path:
        print("logging OpenMV L2FT telemetry to %s" % telemetry_csv_path)
    try:
        while True:
            now = time.monotonic()
            poll_telemetry()
            pygame.event.pump()
            check_joystick_events()
            try:
                check_joystick_attached()
                axes, buttons = read_gamepad(joystick, mapping)
            except pygame.error as e:
                fail_closed("gamepad read failed: %s" % e)
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
                send_packet(armed, roll, pitch, yaw_rate, thrust)
                next_send += period
                if next_send < now - period:
                    next_send = now + period
                last = (roll, pitch, yaw_rate, thrust, armed)

            if now >= next_print:
                row = telemetry_stats["last_row"]
                if row is None:
                    print(
                        "seq=%d armed=%d roll=%+.3f pitch=%+.3f yaw_rate=%+.3f thrust_g=%.3f telemetry=none"
                        % (seq, last[4], last[0], last[1], last[2], last[3])
                    )
                else:
                    print(
                        "seq=%d armed=%d roll=%+.3f pitch=%+.3f yaw_rate=%+.3f thrust_g=%.3f "
                        "telem=%d lost=%d dt_us=%d delay_us=%d a=%+.2f,%+.2f,%+.2f,%+.2f wz=%+.2f,%+.2f,%+.2f"
                        % (
                            seq, last[4], last[0], last[1], last[2], last[3],
                            telemetry_stats["count"], telemetry_stats["lost"],
                            row["dt_us"], row["deadline_delay_us"],
                            row["action_0"], row["action_1"], row["action_2"], row["action_3"],
                            row["world_z_x"], row["world_z_y"], row["world_z_z"],
                        )
                    )
                next_print = now + args.print_every

            sleep_s = min(max(next_send - time.monotonic(), 0.0), 0.01)
            if sleep_s:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        send_disarm_burst("interrupted")
        pass
    finally:
        poll_telemetry()
        if telemetry_csv_file is not None:
            telemetry_csv_file.flush()
            telemetry_csv_file.close()
        pygame.quit()


if __name__ == "__main__":
    main()

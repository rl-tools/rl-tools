import gc
import os
import time

import csi
import image
import machine


CAPTURE_HZ = 10
TICK_US = 1_000_000 // CAPTURE_HZ

WIDTH = 80
HEIGHT = 50
OUTPUT_ROOT = "/flash/capture"
MAX_FRAMES = 100
SAVE_FORMAT = "jpg"
JPEG_QUALITY = 90

CAMERA_FRAMEBUFFERS = 3
CAMERA_SENSOR_FPS = 200
CAMERA_EXPOSURE_US = 500
CAMERA_GAIN_DB = 50
WARMUP_FRAMES = 20
LAG_WARN_US = 1_000
BUTTON_PIN = "SW"

LED_READY_MS = 1000
LED_DONE_MS = 1000


def print_mem(label):
    gc.collect()
    print("%-18s free=%d alloc=%d" % (label, gc.mem_free(), gc.mem_alloc()))


def ensure_dir(path):
    try:
        os.stat(path)
        return True
    except OSError:
        try:
            os.mkdir(path)
            return True
        except OSError as e:
            print("mkdir failed:", path, e)
            return False


def prepare_capture_dir():
    if ensure_dir(OUTPUT_ROOT):
        output_root = OUTPUT_ROOT
    else:
        raise RuntimeError("no capture output directory available")

    for name in os.listdir(output_root):
        if (name.startswith("frame_") or
                name == "index.csv" or
                name == "meta.txt" or
                name == "stats.txt"):
            path = output_root + "/" + name
            try:
                os.remove(path)
            except OSError as e:
                print("remove failed:", path, e)
        try:
            gc.collect()
        except OSError:
            pass
    return output_root


class StatusLed:
    def __init__(self):
        self.led = None
        for idx in (1, 2, 3):
            try:
                self.led = machine.LED(idx)
                break
            except Exception:
                pass
        if self.led is None:
            for name in ("LED_BLUE", "LED_GREEN", "LED_RED", "LED"):
                try:
                    self.led = machine.Pin(name, machine.Pin.OUT)
                    break
                except Exception:
                    pass

    def on(self):
        if self.led is None:
            return
        try:
            self.led.on()
        except Exception:
            self.led.value(1)

    def off(self):
        if self.led is None:
            return
        try:
            self.led.off()
        except Exception:
            self.led.value(0)

    def pulse(self, ms):
        self.on()
        time.sleep_ms(ms)
        self.off()


def make_button():
    return machine.Pin(BUTTON_PIN, machine.Pin.IN, machine.Pin.PULL_UP)


def wait_for_button_press(button, led):
    print("waiting for side button")
    led.pulse(LED_READY_MS)
    last = button.value()
    while True:
        cur = button.value()
        if last == 1 and cur == 0:
            time.sleep_ms(30)
            while button.value() == 0:
                time.sleep_ms(10)
            return
        last = cur
        time.sleep_ms(10)


def configure_camera(csi0):
    csi0.reset()
    csi0.pixformat(csi.RGB565)
    csi0.framesize(csi.QVGA)
    csi0.framerate(CAMERA_SENSOR_FPS)
    csi0.auto_exposure(False, exposure_us=CAMERA_EXPOSURE_US)
    csi0.auto_gain(False, gain_db=CAMERA_GAIN_DB)
    csi0.auto_rotation(False)
    csi0.hmirror(False)
    csi0.vflip(False)
    csi0.transpose(False)
    csi0.auto_blc(False)
    csi0.auto_whitebal(True)


def write_text(path, text):
    f = open(path, "w")
    try:
        f.write(text)
    finally:
        f.close()


def write_ppm(path, rgb):
    f = open(path, "wb")
    try:
        f.write(("P6\n%d %d\n255\n" % (WIDTH, HEIGHT)).encode())
        f.write(rgb)
    finally:
        f.close()


def write_rgb888(path, rgb):
    f = open(path, "wb")
    try:
        f.write(rgb)
    finally:
        f.close()


def write_jpeg(path, data):
    f = open(path, "wb")
    try:
        f.write(data)
    finally:
        f.close()


def compress_jpeg(img):
    try:
        return img.compress(copy=True, quality=JPEG_QUALITY)
    except TypeError:
        tmp = img.copy()
        try:
            return tmp.compress(quality=JPEG_QUALITY)
        except TypeError:
            return tmp.compress()


def jpeg_bytes(img):
    try:
        return bytes(img.bytearray())
    except Exception:
        return bytes(img)


def frame_name(frame_i):
    if SAVE_FORMAT == "jpg" or SAVE_FORMAT == "jpeg":
        return "frame_%06d.jpg" % frame_i
    if SAVE_FORMAT == "ppm":
        return "frame_%06d.ppm" % frame_i
    if SAVE_FORMAT == "rgb888":
        return "frame_%06d.rgb" % frame_i
    raise RuntimeError("unsupported SAVE_FORMAT=%r" % SAVE_FORMAT)


def write_frame(path, data, rgb):
    if SAVE_FORMAT == "jpg" or SAVE_FORMAT == "jpeg":
        write_jpeg(path, data)
    elif SAVE_FORMAT == "ppm":
        write_ppm(path, rgb)
    else:
        write_rgb888(path, rgb)


def write_capture_outputs(capture_dir, meta, frames, rows):
    write_text(capture_dir + "/meta.txt", meta)
    sum_write_us = 0
    max_write_us = 0
    for i in range(len(frames)):
        name, data, rgb = frames[i]
        t0 = time.ticks_us()
        write_frame(capture_dir + "/" + name, data, rgb)
        write_us = time.ticks_diff(time.ticks_us(), t0)
        rows[i].append(write_us)
        sum_write_us += write_us
        if write_us > max_write_us:
            max_write_us = write_us

    index = open(capture_dir + "/index.csv", "w")
    try:
        index.write(
            "frame,path,ticks_us,start_lag_us,snapshot_us,resize_us,compress_us,"
            "capture_us,capture_end_lag_us,capture_missed_deadlines,encoded_bytes,write_us\n"
        )
        for row in rows:
            index.write("%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n" % tuple(row))
    finally:
        index.close()
    return sum_write_us, max_write_us


def build_meta():
    if SAVE_FORMAT == "jpg" or SAVE_FORMAT == "jpeg":
        format_desc = "jpeg"
        bytes_per_frame = -1
    elif SAVE_FORMAT == "rgb888":
        format_desc = "raw_rgb888"
        bytes_per_frame = WIDTH * HEIGHT * 3
    elif SAVE_FORMAT == "ppm":
        format_desc = "ppm_p6_rgb888"
        bytes_per_frame = WIDTH * HEIGHT * 3 + len(("P6\n%d %d\n255\n" % (WIDTH, HEIGHT)).encode())
    else:
        raise RuntimeError("unsupported SAVE_FORMAT=%r" % SAVE_FORMAT)

    meta = (
        "format=%s\n"
        "width=%d\n"
        "height=%d\n"
        "channels=3\n"
        "channel_order=RGB\n"
        "bytes_per_frame=%d\n"
        "jpeg_quality=%d\n"
        "capture_hz=%d\n"
        "tick_us=%d\n"
        "camera_sensor_fps=%d\n"
        "camera_exposure_us=%d\n"
        "camera_gain_db=%d\n"
        "camera_pixformat=RGB565\n"
        "camera_framesize=QVGA\n"
        "camera_auto_rotation=false\n"
        "camera_hmirror=false\n"
        "camera_vflip=false\n"
        "camera_transpose=false\n"
        "camera_auto_blc=false\n"
        "camera_auto_whitebal=true\n"
        "max_frames=%d\n"
    ) % (
        format_desc, WIDTH, HEIGHT, bytes_per_frame, JPEG_QUALITY, CAPTURE_HZ, TICK_US,
        CAMERA_SENSOR_FPS, CAMERA_EXPOSURE_US, CAMERA_GAIN_DB, MAX_FRAMES
    )
    return meta


def capture_episode(csi0, scaled, rgb, draw_hint, led):
    frame_i = 0
    frames = []
    rows = []
    next_deadline = time.ticks_us()
    sum_compress_us = 0
    sum_capture_us = 0
    max_compress_us = 0
    max_capture_us = 0
    max_start_lag_us = 0
    max_capture_end_lag_us = 0
    behind_start_count = 0
    behind_capture_end_count = 0
    capture_missed_deadline_count = 0
    print("capturing %d x %d %s at %d Hz to RAM; MAX_FRAMES=%d" %
          (WIDTH, HEIGHT, SAVE_FORMAT, CAPTURE_HZ, MAX_FRAMES))
    led.on()
    while MAX_FRAMES <= 0 or frame_i < MAX_FRAMES:
        now = time.ticks_us()
        start_lag_us = time.ticks_diff(now, next_deadline)
        if start_lag_us < 0:
            led.off()
            time.sleep_us(-start_lag_us)
            led.on()
            now = time.ticks_us()
            start_lag_us = time.ticks_diff(now, next_deadline)
        if start_lag_us > LAG_WARN_US:
            behind_start_count += 1
            if start_lag_us > max_start_lag_us:
                max_start_lag_us = start_lag_us

        t0 = now
        img = csi0.snapshot()
        t_snapshot = time.ticks_us()
        scaled.draw_image(img, 0, 0, hint=draw_hint)
        t_resize = time.ticks_us()
        encoded = scaled
        encoded_bytes = -1
        if SAVE_FORMAT == "jpg" or SAVE_FORMAT == "jpeg":
            encoded = compress_jpeg(scaled)
            encoded = jpeg_bytes(encoded)
            encoded_bytes = len(encoded)
        elif rgb is not None:
            scaled.to_ndarray(dtype="B", buffer=rgb)
            encoded_bytes = len(rgb)
        t_compress = time.ticks_us()

        name = frame_name(frame_i)
        if SAVE_FORMAT == "jpg" or SAVE_FORMAT == "jpeg":
            frames.append((name, encoded, None))
        else:
            frames.append((name, None, bytes(rgb)))

        snapshot_us = time.ticks_diff(t_snapshot, t0)
        resize_us = time.ticks_diff(t_resize, t_snapshot)
        compress_us = time.ticks_diff(t_compress, t_resize)
        capture_us = time.ticks_diff(t_compress, t0)
        frame_next_deadline = time.ticks_add(next_deadline, TICK_US)
        capture_end_lag_us = time.ticks_diff(t_compress, frame_next_deadline)
        if capture_end_lag_us > LAG_WARN_US:
            behind_capture_end_count += 1
            if capture_end_lag_us > max_capture_end_lag_us:
                max_capture_end_lag_us = capture_end_lag_us
        if compress_us > max_compress_us:
            max_compress_us = compress_us
        if capture_us > max_capture_us:
            max_capture_us = capture_us
        sum_compress_us += compress_us
        sum_capture_us += capture_us

        missed_deadlines = 0
        next_deadline = frame_next_deadline
        while time.ticks_diff(t_compress, next_deadline) >= 0:
            next_deadline = time.ticks_add(next_deadline, TICK_US)
            missed_deadlines += 1
        capture_missed_deadline_count += missed_deadlines
        rows.append([
            frame_i, name, t0, start_lag_us, snapshot_us, resize_us,
            compress_us, capture_us, capture_end_lag_us,
            missed_deadlines, encoded_bytes
        ])

        if frame_i % 2 == 0:
            led.off()
        else:
            led.on()
        frame_i += 1
    led.off()

    stats_head = (
        "frames=%d\n"
        "avg_compress_us=%d\n"
        "max_compress_us=%d\n"
        "avg_capture_us=%d\n"
        "max_capture_us=%d\n"
        "max_start_lag_us=%d\n"
        "max_capture_end_lag_us=%d\n"
        "behind_start_count=%d\n"
        "behind_capture_end_count=%d\n"
        "capture_missed_deadline_count=%d\n"
    ) % (
        frame_i, sum_compress_us // frame_i, max_compress_us,
        sum_capture_us // frame_i, max_capture_us,
        max_start_lag_us, max_capture_end_lag_us,
        behind_start_count, behind_capture_end_count,
        capture_missed_deadline_count
    )
    return frames, rows, stats_head


def save_episode(capture_dir, meta, frames, rows, stats_head, led):
    print("captured %d frames in RAM; writing to %s" %
          (len(frames), capture_dir))
    led.on()
    t_write_all = time.ticks_us()
    sum_write_us, max_write_us = write_capture_outputs(capture_dir, meta, frames, rows)
    write_all_us = time.ticks_diff(time.ticks_us(), t_write_all)
    led.off()
    n = len(frames)
    stats = stats_head + (
        "avg_post_write_us=%d\n"
        "max_post_write_us=%d\n"
        "post_write_all_us=%d\n"
    ) % (sum_write_us // n, max_write_us, write_all_us)
    write_text(capture_dir + "/stats.txt", stats)
    print(stats)
    print("done: frames=%d dir=%s" % (n, capture_dir))


def run():
    print_mem("boot")
    capture_dir = prepare_capture_dir()
    print("capture dir:", capture_dir)

    led = StatusLed()
    button = make_button()

    csi0 = csi.CSI()
    configure_camera(csi0)
    csi0.framebuffers(CAMERA_FRAMEBUFFERS)
    print("camera %dx%d fps=%d exposure_us=%d gain_db=%d" %
          (csi0.width(), csi0.height(), CAMERA_SENSOR_FPS,
           CAMERA_EXPOSURE_US, CAMERA_GAIN_DB))

    for _ in range(WARMUP_FRAMES):
        csi0.snapshot()

    scaled = image.Image(WIDTH, HEIGHT, image.RGB565)
    rgb = None if SAVE_FORMAT in ("jpg", "jpeg") else bytearray(WIDTH * HEIGHT * 3)
    draw_hint = image.BILINEAR | image.SCALE_ASPECT_IGNORE
    meta = build_meta()

    while True:
        wait_for_button_press(button, led)
        capture_dir = prepare_capture_dir()
        frames, rows, stats_head = capture_episode(csi0, scaled, rgb, draw_hint, led)
        save_episode(capture_dir, meta, frames, rows, stats_head, led)
        frames = None
        rows = None
        gc.collect()
        led.pulse(LED_DONE_MS)


if __name__ == "__main__":
    run()

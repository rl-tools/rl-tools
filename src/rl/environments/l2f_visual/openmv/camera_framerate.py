import time
import csi

N = 200

csi0 = csi.CSI()
csi0.reset()
csi0.pixformat(csi.RGB565)
csi0.framesize(csi.QVGA)
csi0.framerate(200)
csi0.auto_exposure(False, exposure_us=500)
csi0.auto_gain(False, gain_db=50)
csi0.auto_rotation(False)
csi0.auto_blc(False)

for _ in range(100):
    csi0.snapshot()

r, g, b = csi0.rgb_gain_db()       # read what AWB learned
print(r, g, b)
csi0.auto_whitebal(False, rgb_gain_db=(r, g, b))   # lock it in

t0 = time.ticks_us()
for _ in range(N):
    csi0.snapshot()
dt = time.ticks_diff(time.ticks_us(), t0)
print("%d frames in %d us -> %.2f fps" % (N, dt, 1e6 * N / dt))


while True:
    csi0.snapshot()

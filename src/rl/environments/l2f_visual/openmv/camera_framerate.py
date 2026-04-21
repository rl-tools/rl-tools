import time
import csi

N = 200

csi0 = csi.CSI()
csi0.reset()
csi0.pixformat(csi.RGB565)
csi0.framesize(csi.QVGA)
csi0.framerate(200)
csi0.auto_exposure(False, exposure_us=2000)

for _ in range(20):
    csi0.snapshot()

t0 = time.ticks_us()
for _ in range(N):
    csi0.snapshot()
dt = time.ticks_diff(time.ticks_us(), t0)
print("%d frames in %d us -> %.2f fps" % (N, dt, 1e6 * N / dt))

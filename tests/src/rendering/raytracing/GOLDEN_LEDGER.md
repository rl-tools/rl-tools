# Raytracing golden ledger

History of the golden corpus versions pinned in `golden_manifest.h`.

## procthor-train-1

| version | RLtools commit | test-data commit | reason |
|---|---|---|---|
| 2 | _pending_ | _pending publication_ | e13d6b037 imports glTF materials per spec (absent `metallicFactor` is 1, so the pose-11 statue and the pose-04 fixture turn metallic); the corpus becomes view x state x output under `rendering/raytracing/goldens/procthor-train-1/<view>/<state>/<output>/` with a manifest, `view.json` and goldens named by generation (`golden-v00` for poses 00-03, `golden-v01`, `golden-v02`); new cases: normals, segmentation, two-instant flow, dynamic motion blur, the camera motion gains a yaw, the states `overlays`, `camera-overlays`, `objects` and `camera-objects` add a shared cube plus one cube private to each view; v1 was rendered on another host; built against Assimp 5.3.1 it reproduces here up to GPU noise (RGB MAD at most 0.04), the host's Assimp 6 changes textured metals |
| 1 | 4b4410607 (2026-08-02) | cb62bfcfd9bd (2026-08-02) | 93217a3a9 replaces the four-pose grid with twelve per-pose directories; 4b4410607 reads metallic/roughness factors from the GLB JSON instead of Assimp, whose defaults vary by version |
| 0 | 054d42b26 (2026-08-02) | fd90f9c61641 (2026-08-02) | initial corpus, four poses; `golden-v00` of poses 04-11 rendered with 054d42b26 against Assimp 5.3.1 on the v2 host on 2026-09-30 (that Assimp reproduces the published four poses to host noise, RGB MAD at most 0.04; the host's Assimp 6 does not) |

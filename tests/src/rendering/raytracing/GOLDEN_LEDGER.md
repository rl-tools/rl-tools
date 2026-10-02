# Raytracing golden ledger

History of the golden corpus versions pinned in `golden_manifest.h`.

## procthor-train-1

| version | RLtools commit | test-data commit | reason |
|---|---|---|---|
| 4 | _pending implementation commit_ | _pending publication_ | separate instance and semantic segmentation; converter v5 exports AI2THOR-Hab taxonomy and root labels in scene `conta:75d1d67a17ecd16bc3a214baa895240da8060df7`; existing outputs retained, generic CPU reference |
| 3 | cbf846a414ae633a490a2239f11536a6b8da6a75 (2026-10-02) | e791a95d8fc12895fbf656716b2389baa2779bf5 (2026-10-02) | generic CPU becomes the reference; incorporates the emissive-strength/unlit fixes, corrected ProcTHOR light assignments, asset-owned ambient lighting (`5e3b99e50`) and converter validity fixes (`0e810278b`); scene `conta:5c53061c721fa98f93e6886925e41e975c577f3c`; generation environment recorded in the manifest |
| 2 | _pending_ | _pending publication_ | e13d6b037 imports glTF materials per spec (absent `metallicFactor` is 1, so the pose-11 statue and the pose-04 fixture turn metallic); the corpus becomes view x state x output under `rendering/raytracing/goldens/procthor-train-1/<view>/<state>/<output>/` with a manifest, `view.json` and goldens named by generation (`golden-v00` for poses 00-03, `golden-v01`, `golden-v02`); new cases: normals, segmentation, two-instant flow, dynamic motion blur, the camera motion gains a yaw, the states `overlays`, `camera-overlays`, `objects` and `camera-objects` add a shared cube plus one cube private to each view; v1 was rendered on another host; built against Assimp 5.3.1 it reproduces here up to GPU noise (RGB MAD at most 0.04), the host's Assimp 6 changes textured metals |
| 1 | 4b4410607 (2026-08-02) | cb62bfcfd9bd (2026-08-02) | 93217a3a9 replaces the four-pose grid with twelve per-pose directories; 4b4410607 reads metallic/roughness factors from the GLB JSON instead of Assimp, whose defaults vary by version |
| 0 | 054d42b26 (2026-08-02) | fd90f9c61641 (2026-08-02) | initial corpus, four poses; `golden-v00` of poses 04-11 rendered with 054d42b26 against Assimp 5.3.1 on the v2 host on 2026-09-30 (that Assimp reproduces the published four poses to host noise, RGB MAD at most 0.04; the host's Assimp 6 does not) |

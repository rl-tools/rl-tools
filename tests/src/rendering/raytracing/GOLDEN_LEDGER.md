# Raytracing golden ledger

History of the golden corpus versions pinned in `golden_manifest.h`.

## procthor_static_scene

| version | RLtools commit | test-data commit | reason |
|---|---|---|---|
| 2 | 853df253b (2026-09-23) | _pending publication_ | e13d6b037 imports glTF materials per spec (absent `metallicFactor` is 1, so the pose-11 statue and the pose-04 fixture turn metallic); new cases: dynamic motion blur, normals, segmentation, two-instant flow; layout moves to `rendering/raytracing/goldens/procthor-train-1/<view>/<modality>/` with a manifest, `view.json` and the v1 goldens archived as `golden-v01`; v1 was rendered on another host and is not bit-reproducible here |
| 1 | 4b4410607 (2026-08-02) | cb62bfcfd9bd (2026-08-02) | 93217a3a9 replaces the four-pose grid with twelve per-pose directories; 4b4410607 reads metallic/roughness factors from the GLB JSON instead of Assimp, whose defaults vary by version |
| 0 | 054d42b26 (2026-08-02) | fd90f9c61641 (2026-08-02) | initial corpus |

## overlay

| version | RLtools commit | test-data commit | reason |
|---|---|---|---|
| 1 | 853df253b (2026-09-23) | _pending publication_ | initial corpus |

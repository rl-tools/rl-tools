```
hf download hssd/ai2thor-hab --repo-type dataset --local-dir data/ai2thor-hab
```
```
./build/src/rendering/procthor2glb/procthor2glb src/rendering/procthor2glb/data/ai2thor-hab/ai2thor-hab/configs/scenes/ProcTHOR/1/ProcTHOR-Test-0.scene_instance.json -o ProcTHOR-Test-0-new.glb --normalize
```

### Bulk
```
 find src/rendering/procthor2glb/data/ai2thor-hab/ai2thor-hab/configs/scenes/ProcTHOR/* | grep Train | sort -V | head -200 | xargs -I{} -P 16 bash -c './cmake-build-release/src/rendering/procthor2glb/procthor2glb {} -o  src/rendering/procthor2glb/data/ai2thor-hab/glb/$(basename {} .scene_instance.json).glb --normalize'
```
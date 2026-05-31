```
hf download hssd/ai2thor-hab --repo-type dataset --local-dir data/ai2thor-hab
```
```
./build/src/rendering/procthor2glb/procthor2glb src/rendering/procthor2glb/data/ai2thor-hab/ai2thor-hab/configs/scenes/ProcTHOR/1/ProcTHOR-Test-0.scene_instance.json -o ProcTHOR-Test-0-new.glb --normalize
```

### HSSD
```
mkdir -p /home/jonas/git/hssd-hab/glb
./build/src/rendering/procthor2glb/procthor2glb /home/jonas/git/hssd-hab/scenes/105515430_173104494.scene_instance.json --hssd --normalize -o /home/jonas/git/hssd-hab/glb/105515430_173104494.glb
```

HSSD articulated scene files can be passed with `--hssd` as well, but `articulated_object_instances` are skipped because URDF joint conversion is not implemented.

```
./build/src/rendering/procthor2glb/procthor2glb /home/jonas/git/hssd-hab/scenes-articulated/105515430_173104494.scene_instance.json --hssd --normalize -o /home/jonas/git/hssd-hab/glb/105515430_173104494_articulated_static.glb
```

If an HSSD scene references a Habitat lighting setup through `default_lighting`, `--hssd` imports supported punctual lights into `KHR_lights_punctual` and preserves unsupported lighting records in GLB extras. An empty or omitted `default_lighting` value uses Habitat's built-in default lights; use `--hssd-lighting` to import an explicit lighting JSON file instead.

```
./build/src/rendering/procthor2glb/procthor2glb /home/jonas/git/hssd-hab/scenes/105515430_173104494.scene_instance.json --hssd --hssd-lighting /path/to/lighting.json --normalize -o /home/jonas/git/hssd-hab/glb/105515430_173104494_lit.glb
```

### Bulk
```
 find src/rendering/procthor2glb/data/ai2thor-hab/ai2thor-hab/configs/scenes/ProcTHOR/* | grep Train | sort -V | head -200 | xargs -I{} -P 16 bash -c './cmake-build-release/src/rendering/procthor2glb/procthor2glb {} -o  src/rendering/procthor2glb/data/ai2thor-hab/glb/$(basename {} .scene_instance.json).glb --normalize'
```

### HSSD Bulk
```
mkdir -p /home/jonas/git/hssd-hab/glb
find /home/jonas/git/hssd-hab/scenes -name '*.scene_instance.json' | sort -V | xargs -I{} -P 5 bash -c './build/src/rendering/procthor2glb/procthor2glb "$1" --hssd --normalize -o "/home/jonas/git/hssd-hab/glb/$(basename "$1" .scene_instance.json).glb"' _ {}
```

```
hf download hssd/ai2thor-hab --repo-type dataset --local-dir data/ai2thor-hab
```
```
./build/src/hyperlux/procthor2glb/procthor2glb src/hyperlux/procthor2glb/data/ai2thor-hab/ai2thor-hab/configs/scenes/ProcTHOR/1/ProcTHOR-Test-0.scene_instance.json -o ProcTHOR-Test-0-new.glb --normalize
```

ProcTHOR exports carry `scenes[].extras.rl_tools.environment` with linear RGB ambient `[0.6, 0.6, 0.6]` and a solid black background, which RLtools adopts when loading the complete scene. HSSD exports use ambient `[0.7, 0.7, 0.7]`, which matches their median image luminance to ProcTHOR's.

### HSSD
```
mkdir -p /home/jonas/git/hssd-hab/glb
./build/src/hyperlux/procthor2glb/procthor2glb /home/jonas/git/hssd-hab/scenes/105515430_173104494.scene_instance.json --hssd --normalize -o /home/jonas/git/hssd-hab/glb/105515430_173104494.glb
```

HSSD articulated scene files can be passed with `--hssd` as well, but `articulated_object_instances` are skipped because URDF joint conversion is not implemented.

```
./build/src/hyperlux/procthor2glb/procthor2glb /home/jonas/git/hssd-hab/scenes-articulated/105515430_173104494.scene_instance.json --hssd --normalize -o /home/jonas/git/hssd-hab/glb/105515430_173104494_articulated_static.glb
```

If an HSSD scene references a Habitat lighting setup through `default_lighting`, `--hssd` imports supported punctual lights into `KHR_lights_punctual` and preserves unsupported lighting records in GLB extras. An empty or omitted `default_lighting` value uses Habitat's built-in default lights; use `--hssd-lighting` to import an explicit lighting JSON file instead.

```
./build/src/hyperlux/procthor2glb/procthor2glb /home/jonas/git/hssd-hab/scenes/105515430_173104494.scene_instance.json --hssd --hssd-lighting /path/to/lighting.json --normalize -o /home/jonas/git/hssd-hab/glb/105515430_173104494_lit.glb
```

### Bulk
```
 find src/hyperlux/procthor2glb/data/ai2thor-hab/ai2thor-hab/configs/scenes/ProcTHOR/* | grep Train | sort -V | head -200 | xargs -I{} -P 16 bash -c './cmake-build-release/src/hyperlux/procthor2glb/procthor2glb {} -o  src/hyperlux/procthor2glb/data/ai2thor-hab/glb/$(basename {} .scene_instance.json).glb --normalize'
```

### HSSD Bulk
```
mkdir -p /home/jonas/git/hssd-hab/glb
find /home/jonas/git/hssd-hab/scenes -name '*.scene_instance.json' | sort -V | xargs -I{} -P 5 bash -c './build/src/hyperlux/procthor2glb/procthor2glb "$1" --hssd --normalize -o "/home/jonas/git/hssd-hab/glb/$(basename "$1" .scene_instance.json).glb"' _ {}
```

AI2THOR-Hab conversion exports `scenes[].extras.rl_tools.segmentation` with `taxonomy: "ai2thor-hab"` and a `classes` name-to-ID map from `configs/object_semantic_id_mapping.json`. Each root carries `extras.rl_tools.segmentation_class`: movable objects use their config's `semantic_id`, structural roots resolve their authored `type`; missing categories (including Ceiling) map to Undefined (`0`). IDs are never inferred from asset names. The conversion cache includes the taxonomy, stage content and referenced object labels.

HSSD conversion exports `taxonomy: "hssd-hab"` with the classes of `semantics/hssd-hab_semantic_lexicon.json`. An object root's class is its template's category in `metadata/hssd_obj_semantics_condensed.csv`, else its `semantics/objects.csv` `main_category`, matched to a lexicon class by exact name, else `unknown` (`0`); decomposed parts (`<hash>_part_<n>`) fall back to the rows of `<hash>`. The object configs' `semantic_id` is ignored: below 25 it follows a stale ID space. Stage roots resolve only an authored `type`, which HSSD stages lack. Each object root carries `extras.hssd` with the join keys `template` and `instance` (its `object_instances` index) and `categories`, the non-empty `wnsynsetkey`, `main_category` and `super_category` (`objects.csv`) and `condensed_category`, `primary_category` and `pickable` (condensed table). `scenes[].extras.hssd` holds `scene` and the verbatim `region_annotations` of `semantics/scenes/<id>.semantic_config.json`, in HSSD's Y-up frame. A merged object without an object config, or a scene without a semantic config, fails the conversion. Identical bufferView bytes are stored once. Generic GLB loading interprets this annotation without dataset-specific rules; C++ consumers use `Mode<mode::hyperlux::SceneRoots<>>` and request `outputs::Segmentation<outputs::format::UInt32, outputs::labels::Semantic>` in `Config::OUTPUTS`.

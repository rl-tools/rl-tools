// WebGPU device programs for the RLtools raytracing renderer: one module, one entry point per
// output group (main_rgb / main_auxiliary / main_collision / main_flow_forward).
// The auxiliary outputs share their shutter-close pixel-center hit. Shading mirrors
// backends/vulkan/device.comp (which mirrors backends/metal/device.metal and
// backends/optix/device_impl.h) — keep the shading logic in sync; the golden-image parity suite
// is the drift detector. Standard WebGPU has no ray query, so the hardware traversal is replaced
// by the software two-level BVH of backends/generic/operations_generic.h (same node layout,
// same traversal order, same Moeller-Trumbore epsilons); textures are sampled manually from a
// packed RGBA8 buffer with the generic backend's bilinear + per-texel-sRGB filter.
// Struct layouts (LaunchParams, MeshRecord, InstanceData, BVHNode) and binding
// indices must match include/rl_tools/rendering/raytracing/backends/webgpu/context.h.

override fc_srgb_output: bool = false;
override fc_motion_blur: bool = false;
override fc_motion_samples: i32 = 1;
override fc_aa_grid: i32 = 1;
override fc_checker_background: bool = false;
override fc_load_textures: bool = false;
override fc_normal_shading: bool = false;
override fc_metallic_reflections: bool = false;
override fc_pbr_shading: bool = false;
override fc_punctual_light_shadows: bool = false;
override fc_overlay_count: i32 = 0;
override fc_output_mask: u32 = 1u;
override fc_fuse_auxiliary: bool = false;
override fc_dynamic_motion_blur: bool = false;
override fc_num_overlays: u32 = 0u;
override fc_overlay_capacity: u32 = 0u;

const ABSENT: u32 = 0xFFFFFFFFu;
const TRAVERSAL_STACK_SIZE: u32 = 48u;
const PI: f32 = 3.14159265;

struct LaunchParams{
    fb_width: u32,
    fb_height: u32,
    cam_width: u32,
    cam_height: u32,
    grid_cols: u32,
    num_cameras: u32,
    num_probes: u32,
    num_scene_lights: u32,
    max_dist: f32,
    ambient_color: array<f32, 3>,
    miss_color_0: array<f32, 3>,
    miss_color_1: array<f32, 3>,
    first_overlay_instance: u32,
    num_overlay_slots: u32,
    triangle_mesh_offset: u32,
    triangle_local_offset: u32,
    object_records_offset: u32,
    tlas_node_offset: u32,
    tlas_node_count: u32,
    tlas_primitive_offset: u32,
    srgb_lut_offset: u32,
    mesh_records_offset: u32,
    scene_lights_offset: u32,
    instance_classes_offset: u32,
    overlay_node_offset: u32,
    attachments_offset: u32,
    overlay_meta_offset: u32,
    overlay_primitives_offset: u32,
    flow_deltas_offset: u32,
    probe_directions_offset: u32,
    cameras_offset: u32, // shutter-close cameras, then the shutter-open set when the spec has a camera pair
    cameras_previous_offset: u32, // flow: the previous instant's camera set
    output_offsets: array<u32, 19>,
    output_scales: array<f32, 19>,
}

const OUTPUT_RGB_RGBA8: u32 = 0u;
const OUTPUT_RGB_FLOAT32: u32 = 1u;
const OUTPUT_DEPTH_FLOAT32: u32 = 2u;
const OUTPUT_DEPTH_RGBA8: u32 = 3u;
const OUTPUT_SEGMENTATION_UINT32: u32 = 4u;
const OUTPUT_SEGMENTATION_RGBA8: u32 = 5u;
const OUTPUT_SEMANTIC_UINT32: u32 = 6u;
const OUTPUT_SEMANTIC_RGBA8: u32 = 7u;
const OUTPUT_NORMALS_FLOAT32: u32 = 8u;
const OUTPUT_NORMALS_RGBA8: u32 = 9u;
const OUTPUT_FLOW_FLOAT32: u32 = 10u;
const OUTPUT_FLOW_RGBA8: u32 = 11u;
const OUTPUT_FLOW_STATUS_UINT32: u32 = 12u;
const OUTPUT_FLOW_STATUS_RGBA8: u32 = 13u;
const OUTPUT_FLOW_FORWARD_FLOAT32: u32 = 14u;
const OUTPUT_FLOW_FORWARD_RGBA8: u32 = 15u;
const OUTPUT_FLOW_FORWARD_STATUS_UINT32: u32 = 16u;
const OUTPUT_FLOW_FORWARD_STATUS_RGBA8: u32 = 17u;
const OUTPUT_PROBES: u32 = 18u;

fn output_enabled(slot: u32) -> bool{
    return (fc_output_mask & (1u << slot)) != 0u;
}

fn hue_to_rgb(hue: f32) -> vec3<f32>{
    let descending = 1.0 - abs((hue - 2.0 * floor(hue * 0.5)) - 1.0);
    switch(u32(hue) % 6u){
        case 0u: {
            return vec3<f32>(1.0, descending, 0.0);
        }
        case 1u: {
            return vec3<f32>(descending, 1.0, 0.0);
        }
        case 2u: {
            return vec3<f32>(0.0, 1.0, descending);
        }
        case 3u: {
            return vec3<f32>(0.0, descending, 1.0);
        }
        case 4u: {
            return vec3<f32>(descending, 0.0, 1.0);
        }
        default: {
            return vec3<f32>(1.0, 0.0, descending);
        }
    }
}
fn pack_color(rgb: vec3<u32>) -> u32{
    return 0xFF000000u | (rgb.z << 16u) | (rgb.y << 8u) | rgb.x;
}
fn segmentation_id_to_rgba(id: u32) -> u32{
    if(id == 0xFFFFFFFFu){
        return 0xFF000000u;
    }
    let scaled = f32(id) * 0.61803398875;
    let hue = (scaled - floor(scaled)) * 6.0;
    return pack_color(vec3<u32>(hue_to_rgb(hue) * 255.0));
}
// Preserve the CPU encoder's two f32 roundings and f64 half-up bias without allowing
// multiplication by 255 to reassociate across the first rounding on fast-math GPUs.
fn normal_channel(n: f32) -> u32{
    let unit_bits = bitcast<u32>(clamp(n, -1.0, 1.0) * 0.5 + 0.5);
    if(unit_bits < 0x3B000000u){
        return 0u;
    }
    let product = ((unit_bits & 0x7FFFFFu) | 0x800000u) * 255u;
    let shift = select(7u, 8u, product >= 0x80000000u);
    let whole = product >> shift;
    let remainder = product & ((1u << shift) - 1u);
    let midpoint = 1u << (shift - 1u);
    let round_up = remainder > midpoint || (remainder == midpoint && (whole & 1u) != 0u);
    let rounded = whole + select(0u, 1u, round_up);
    let integer_shift = 150u - (unit_bits >> 23u) - shift;
    return (rounded + (1u << (integer_shift - 1u))) >> integer_shift;
}
fn normal_to_rgba(normal: vec3<f32>) -> u32{
    return pack_color(vec3<u32>(normal_channel(normal.x), normal_channel(normal.y), normal_channel(normal.z)));
}
fn flow_to_rgba(u: f32, v: f32, magnitude_scale: f32) -> u32{
    if((u == 0.0 && v == 0.0) || magnitude_scale <= 0.0){
        return 0xFFFFFFFFu;
    }
    let magnitude = sqrt(u * u + v * v);
    let saturation = min(magnitude / magnitude_scale, 1.0);
    var angle = atan2(v, u);
    if(u == 0.0){
        angle = select(-1.57079632679489661923, 1.57079632679489661923, v > 0.0);
    }
    let hue = (angle / 6.28318530717958647692 + 0.5) * 6.0;
    let color = hue_to_rgb(hue);
    return pack_color(vec3<u32>((vec3<f32>(1.0) - saturation * (vec3<f32>(1.0) - color)) * 255.0));
}
fn flow_status_to_rgba(status: u32) -> u32{
    switch(status){
        case 0u: {
            return 0xFFFFFFFFu;
        }
        case 1u: {
            return 0xFF000000u;
        }
        case 2u: {
            return 0xFFFF00FFu;
        }
        case 3u: {
            return 0xFFFF4020u;
        }
        case 4u: {
            return 0xFF0090FFu;
        }
        default: {
            return 0xFF0000FFu;
        }
    }
}
fn depth_to_rgba(depth: f32, depth_scale: f32) -> u32{
    let shade = u32(clamp(depth / depth_scale, 0.0, 1.0) * 255.0);
    return pack_color(vec3<u32>(shade));
}

struct Camera{
    pos: array<f32, 3>,
    dir_00: array<f32, 3>,
    dir_du: array<f32, 3>,
    dir_dv: array<f32, 3>,
}

// two vec4 loads instead of eight scalar loads; byte-identical to the host BVHNode
// (bounds_min[3], bounds_max[3], left_or_first, count)
struct BVHNode{
    a: vec4<f32>, // bounds_min.xyz, bounds_max.x
    b: vec4<f32>, // bounds_max.yz, bitcast(left_or_first), bitcast(count)
}
fn bvh_left_or_first(node: BVHNode) -> u32{ return bitcast<u32>(node.b.z); }
fn bvh_count(node: BVHNode) -> u32{ return bitcast<u32>(node.b.w); }

struct TextureRef{
    offset: u32,
    width: u32,
    height: u32,
}

struct MeshRecord{
    index_offset: u32,
    vertex_offset: u32,
    tex_coord_offset: u32,
    normal_offset: u32,
    texture: TextureRef,
    normal_map: TextureRef,
    metallic_roughness_map: TextureRef,
    emissive_map: TextureRef,
    occlusion_map: TextureRef,
    color: array<f32, 3>,
    metallic: f32,
    roughness: f32,
    opacity: f32,
    emissive: array<f32, 3>,
    alpha_cutoff: f32,
    alpha_mode: i32,
    unlit: i32,
}

struct InstanceData{
    object_to_world: array<f32, 12>, // 3x4 row-major [R|t]
    world_to_object: array<f32, 12>,
    object: u32,
    identity: u32,
    padding0: u32,
    padding1: u32,
}

struct SceneLight{
    light_type: i32, // 0=directional, 1=point, 2=spot
    position: array<f32, 3>,
    direction: array<f32, 3>,
    color: array<f32, 3>,
    attenuation_constant: f32,
    attenuation_linear: f32,
    attenuation_quadratic: f32,
    cos_inner_cone: f32,
    cos_outer_cone: f32,
}

// exactly 8 storage buffers (browsers tier maxStorageBuffersPerShaderStage to the spec default
// of 8): the small per-frame inputs (cameras, overlay tables, flow deltas, probe directions)
// are sections of FRAME_INPUTS, all outputs sections of OUTPUTS, and the shading-only mesh
// records a section of SCENE_GEOMETRY, addressed by the LaunchParams element offsets — the
// leaf-ordered TRIANGLES stream keeps its own vec4-typed binding for the hot intersection loop.
// LaunchParams stays in the storage address space: as a uniform, the NVIDIA driver (595.84, via
// naga's SPIR-V) miscompiles the punctual-shadow branch in shade_pbr_local
@group(0) @binding(0) var<storage, read> params: LaunchParams;
@group(0) @binding(1) var<storage, read> scene_geometry: array<u32>;
@group(0) @binding(2) var<storage, read> texture_data: array<u32>; // packed RGBA8, one texel per u32
@group(0) @binding(3) var<storage, read> bvh_nodes: array<BVHNode>; // BLAS slices, scene TLAS, overlay TLAS regions
@group(0) @binding(4) var<storage, read> triangles: array<vec4<f32>>; // leaf-ordered packed: 3 x vec4 per triangle, first w = bitcast global id
@group(0) @binding(5) var<storage, read> instance_data: array<InstanceData>;
@group(0) @binding(6) var<storage, read> frame_inputs: array<u32>;
@group(0) @binding(7) var<storage, read_write> outputs: array<u32>;

fn to_vec3(a: array<f32, 3>) -> vec3<f32>{
    return vec3<f32>(a[0], a[1], a[2]);
}

fn camera_pos(cam: Camera) -> vec3<f32>{ return to_vec3(cam.pos); }
fn camera_dir_00(cam: Camera) -> vec3<f32>{ return to_vec3(cam.dir_00); }
fn camera_dir_du(cam: Camera) -> vec3<f32>{ return to_vec3(cam.dir_du); }
fn camera_dir_dv(cam: Camera) -> vec3<f32>{ return to_vec3(cam.dir_dv); }

fn geometry_u32(index: u32) -> u32{
    return scene_geometry[index];
}
fn geometry_f32(index: u32) -> f32{
    return bitcast<f32>(scene_geometry[index]);
}
fn frame_u32(index: u32) -> u32{
    return frame_inputs[index];
}
fn frame_f32(index: u32) -> f32{
    return bitcast<f32>(frame_inputs[index]);
}
fn out_set_f32(index: u32, value: f32){
    outputs[index] = bitcast<u32>(value);
}

fn load_camera_at(base: u32) -> Camera{
    var camera: Camera;
    camera.pos = array<f32, 3>(frame_f32(base), frame_f32(base + 1u), frame_f32(base + 2u));
    camera.dir_00 = array<f32, 3>(frame_f32(base + 3u), frame_f32(base + 4u), frame_f32(base + 5u));
    camera.dir_du = array<f32, 3>(frame_f32(base + 6u), frame_f32(base + 7u), frame_f32(base + 8u));
    camera.dir_dv = array<f32, 3>(frame_f32(base + 9u), frame_f32(base + 10u), frame_f32(base + 11u));
    return camera;
}
fn camera_previous(cam_idx: i32) -> Camera{ return load_camera_at(params.cameras_previous_offset + u32(cam_idx) * 12u); }
fn load_camera(index: u32) -> Camera{
    let base = params.cameras_offset + index * 12u;
    var camera: Camera;
    camera.pos = array<f32, 3>(frame_f32(base), frame_f32(base + 1u), frame_f32(base + 2u));
    camera.dir_00 = array<f32, 3>(frame_f32(base + 3u), frame_f32(base + 4u), frame_f32(base + 5u));
    camera.dir_du = array<f32, 3>(frame_f32(base + 6u), frame_f32(base + 7u), frame_f32(base + 8u));
    camera.dir_dv = array<f32, 3>(frame_f32(base + 9u), frame_f32(base + 10u), frame_f32(base + 11u));
    return camera;
}
fn camera_close(cam_idx: i32) -> Camera{ return load_camera(u32(cam_idx)); }
fn camera_open(cam_idx: i32) -> Camera{ return load_camera(params.num_cameras + u32(cam_idx)); }

fn load_scene_light(light_i: u32) -> SceneLight{
    let base = params.scene_lights_offset + light_i * 15u;
    var light: SceneLight;
    light.light_type = bitcast<i32>(geometry_u32(base));
    light.position = array<f32, 3>(geometry_f32(base + 1u), geometry_f32(base + 2u), geometry_f32(base + 3u));
    light.direction = array<f32, 3>(geometry_f32(base + 4u), geometry_f32(base + 5u), geometry_f32(base + 6u));
    light.color = array<f32, 3>(geometry_f32(base + 7u), geometry_f32(base + 8u), geometry_f32(base + 9u));
    light.attenuation_constant = geometry_f32(base + 10u);
    light.attenuation_linear = geometry_f32(base + 11u);
    light.attenuation_quadratic = geometry_f32(base + 12u);
    light.cos_inner_cone = geometry_f32(base + 13u);
    light.cos_outer_cone = geometry_f32(base + 14u);
    return light;
}

fn mesh_record_base(mesh: u32) -> u32{ return params.mesh_records_offset + mesh * 31u; }
fn mesh_index_offset(mesh: u32) -> u32{ return geometry_u32(mesh_record_base(mesh)); }
fn mesh_vertex_offset(mesh: u32) -> u32{ return geometry_u32(mesh_record_base(mesh) + 1u); }
fn mesh_tex_coord_offset(mesh: u32) -> u32{ return geometry_u32(mesh_record_base(mesh) + 2u); }
fn mesh_normal_offset(mesh: u32) -> u32{ return geometry_u32(mesh_record_base(mesh) + 3u); }
fn load_texture_ref(base: u32) -> TextureRef{
    var reference: TextureRef;
    reference.offset = geometry_u32(base);
    reference.width = geometry_u32(base + 1u);
    reference.height = geometry_u32(base + 2u);
    return reference;
}
fn load_mesh_record(mesh: u32) -> MeshRecord{
    let base = mesh_record_base(mesh);
    var record: MeshRecord;
    record.index_offset = geometry_u32(base);
    record.vertex_offset = geometry_u32(base + 1u);
    record.tex_coord_offset = geometry_u32(base + 2u);
    record.normal_offset = geometry_u32(base + 3u);
    record.texture = load_texture_ref(base + 4u);
    record.normal_map = load_texture_ref(base + 7u);
    record.metallic_roughness_map = load_texture_ref(base + 10u);
    record.emissive_map = load_texture_ref(base + 13u);
    record.occlusion_map = load_texture_ref(base + 16u);
    record.color = array<f32, 3>(geometry_f32(base + 19u), geometry_f32(base + 20u), geometry_f32(base + 21u));
    record.metallic = geometry_f32(base + 22u);
    record.roughness = geometry_f32(base + 23u);
    record.opacity = geometry_f32(base + 24u);
    record.emissive = array<f32, 3>(geometry_f32(base + 25u), geometry_f32(base + 26u), geometry_f32(base + 27u));
    record.alpha_cutoff = geometry_f32(base + 28u);
    record.alpha_mode = bitcast<i32>(geometry_u32(base + 29u));
    record.unlit = bitcast<i32>(geometry_u32(base + 30u));
    return record;
}

fn transform_point(transform: array<f32, 12>, point: vec3<f32>) -> vec3<f32>{
    return vec3<f32>(
        transform[0] * point.x + transform[1] * point.y + transform[2]  * point.z + transform[3],
        transform[4] * point.x + transform[5] * point.y + transform[6]  * point.z + transform[7],
        transform[8] * point.x + transform[9] * point.y + transform[10] * point.z + transform[11]
    );
}

fn transform_vector(transform: array<f32, 12>, vector: vec3<f32>) -> vec3<f32>{
    return vec3<f32>(
        transform[0] * vector.x + transform[1] * vector.y + transform[2]  * vector.z,
        transform[4] * vector.x + transform[5] * vector.y + transform[6]  * vector.z,
        transform[8] * vector.x + transform[9] * vector.y + transform[10] * vector.z
    );
}

// normals transform with the inverse-transpose: multiply by the world_to_object columns
fn transform_normal(world_to_object: array<f32, 12>, normal: vec3<f32>) -> vec3<f32>{
    return vec3<f32>(
        world_to_object[0] * normal.x + world_to_object[4] * normal.y + world_to_object[8]  * normal.z,
        world_to_object[1] * normal.x + world_to_object[5] * normal.y + world_to_object[9]  * normal.z,
        world_to_object[2] * normal.x + world_to_object[6] * normal.y + world_to_object[10] * normal.z
    );
}

struct Hit{
    t: f32,
    u: f32,
    v: f32,
    triangle: u32,
    instance: u32,
    valid: bool,
}

fn fetch_triangle_mesh(triangle: u32) -> u32{
    return geometry_u32(params.triangle_mesh_offset + triangle);
}
fn fetch_triangle_local(triangle: u32) -> u32{
    return geometry_u32(params.triangle_local_offset + triangle);
}
fn fetch_index(mesh: u32, primitive: u32) -> vec3<u32>{
    let base = mesh_index_offset(mesh) + 3u * primitive;
    return vec3<u32>(geometry_u32(base), geometry_u32(base + 1u), geometry_u32(base + 2u));
}
fn fetch_vertex(mesh: u32, index: u32) -> vec3<f32>{
    let base = mesh_vertex_offset(mesh) + 3u * index;
    return vec3<f32>(geometry_f32(base), geometry_f32(base + 1u), geometry_f32(base + 2u));
}
fn fetch_tex_coord(mesh: u32, index: u32) -> vec2<f32>{
    let base = mesh_tex_coord_offset(mesh) + 2u * index;
    return vec2<f32>(geometry_f32(base), geometry_f32(base + 1u));
}
fn fetch_normal(mesh: u32, index: u32) -> vec3<f32>{
    let base = mesh_normal_offset(mesh) + 3u * index;
    return vec3<f32>(geometry_f32(base), geometry_f32(base + 1u), geometry_f32(base + 2u));
}

fn triangle_vertices(triangle: u32) -> array<vec3<f32>, 3>{
    let mesh = fetch_triangle_mesh(triangle);
    let index = fetch_index(mesh, fetch_triangle_local(triangle));
    return array<vec3<f32>, 3>(fetch_vertex(mesh, index.x), fetch_vertex(mesh, index.y), fetch_vertex(mesh, index.z));
}

// Moeller-Trumbore over the leaf-ordered packed stream, no culling; barycentric convention
// matches the other backends (u on the second vertex, v on the third, w0 = 1 - u - v on the
// first); the packed slot's first w component carries the global triangle id for shading
fn intersect_triangle_packed(packed_slot: u32, origin: vec3<f32>, direction: vec3<f32>, t_min: f32, t_max: f32, hit: ptr<function, Hit>) -> bool{
    let p0 = triangles[packed_slot * 3u];
    let p1 = triangles[packed_slot * 3u + 1u];
    let p2 = triangles[packed_slot * 3u + 2u];
    let edge1 = p1.xyz - p0.xyz;
    let edge2 = p2.xyz - p0.xyz;
    let pvec = cross(direction, edge2);
    let det = dot(edge1, pvec);
    if(det > -1e-9 && det < 1e-9){ return false; }
    let inv_det = 1.0 / det;
    let tvec = origin - p0.xyz;
    let u = dot(tvec, pvec) * inv_det;
    if(u < 0.0 || u > 1.0){ return false; }
    let qvec = cross(tvec, edge1);
    let v = dot(direction, qvec) * inv_det;
    if(v < 0.0 || u + v > 1.0){ return false; }
    let t = dot(edge2, qvec) * inv_det;
    if(t <= t_min || t >= t_max){ return false; }
    (*hit).t = t;
    (*hit).u = u;
    (*hit).v = v;
    (*hit).triangle = bitcast<u32>(p0.w);
    (*hit).valid = true;
    return true;
}

// the reciprocal is computed once per traversal, not per node; near-zero components get a
// signed huge value instead of relying on 1/0 = inf (inf semantics are not guaranteed in WGSL)
fn safe_inverse(direction: vec3<f32>) -> vec3<f32>{
    return vec3<f32>(
        select(1.0 / direction.x, select(-1e30, 1e30, direction.x >= 0.0), abs(direction.x) < 1e-12),
        select(1.0 / direction.y, select(-1e30, 1e30, direction.y >= 0.0), abs(direction.y) < 1e-12),
        select(1.0 / direction.z, select(-1e30, 1e30, direction.z >= 0.0), abs(direction.z) < 1e-12)
    );
}

// entry distance for ordered traversal; MISS_DISTANCE on miss
const MISS_DISTANCE: f32 = 1e30;
fn intersect_aabb_distance(node: BVHNode, origin: vec3<f32>, inv_direction: vec3<f32>, t_min: f32, t_max: f32) -> f32{
    let t1 = (node.a.xyz - origin) * inv_direction;
    let t2 = (vec3<f32>(node.a.w, node.b.x, node.b.y) - origin) * inv_direction;
    let t_enter = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), max(min(t1.z, t2.z), t_min));
    let t_exit = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), min(max(t1.z, t2.z), t_max));
    return select(MISS_DISTANCE, t_enter, t_enter <= t_exit);
}

// module-scope traversal stacks: every inlined trace site shares them (traversals only nest
// TLAS -> BLAS), where function-scope arrays cost private memory per inlined copy
var<private> tlas_stack: array<u32, TRAVERSAL_STACK_SIZE>;
var<private> blas_stack: array<u32, TRAVERSAL_STACK_SIZE>;

// BLAS traversal against one instance; the ray is transformed into object space with an
// unnormalized direction, so hit.t stays parameterized in world units and is comparable
// across instances. Ordered: both children are tested at the parent and the nearer subtree is
// descended first, so the best.t cull prunes the far subtree early. any_hit queries (shadow
// and flow visibility rays: is there a hit in (t_min, best.t)?) share the ordered traversal and
// return at the first hit, which near-first ordering reaches early on occluded rays
fn intersect_blas(instance_index: u32, origin_in: vec3<f32>, direction_in: vec3<f32>, inv_direction_in: vec3<f32>, t_min: f32, best: ptr<function, Hit>, any_hit: bool) -> bool{
    // per-field loads: identity instances skip the 96-byte matrix load entirely
    let object = instance_data[instance_index].object;
    let record_base = params.object_records_offset + 4u * object;
    let node_base = geometry_u32(record_base);
    let node_count = geometry_u32(record_base + 1u);
    let first_triangle = geometry_u32(record_base + 2u);
    if(node_count == 0u){ return false; }
    var origin = origin_in;
    var direction = direction_in;
    var inv_direction = inv_direction_in;
    if(instance_data[instance_index].identity == 0u){
        let world_to_object = instance_data[instance_index].world_to_object;
        origin = transform_point(world_to_object, origin_in);
        direction = transform_vector(world_to_object, direction_in);
        inv_direction = safe_inverse(direction);
    }
    if(intersect_aabb_distance(bvh_nodes[node_base], origin, inv_direction, t_min, (*best).t) >= (*best).t){ return false; }
    var stack_pointer = 0u;
    var node_index = 0u;
    var found = false;
    loop{
        let node = bvh_nodes[node_base + node_index];
        let left_or_first = bvh_left_or_first(node);
        let count = bvh_count(node);
        if(count > 0u){
            for(var i = 0u; i < count; i++){
                if(intersect_triangle_packed(first_triangle + left_or_first + i, origin, direction, t_min, (*best).t, best)){
                    (*best).instance = instance_index;
                    found = true;
                    if(any_hit){ return true; }
                }
            }
        }
        else{
            let t_left = intersect_aabb_distance(bvh_nodes[node_base + left_or_first], origin, inv_direction, t_min, (*best).t);
            let t_right = intersect_aabb_distance(bvh_nodes[node_base + left_or_first + 1u], origin, inv_direction, t_min, (*best).t);
            var near_index = left_or_first;
            var near_t = t_left;
            var far_index = left_or_first + 1u;
            var far_t = t_right;
            if(t_right < t_left){
                near_index = left_or_first + 1u;
                near_t = t_right;
                far_index = left_or_first;
                far_t = t_left;
            }
            if(far_t < (*best).t && stack_pointer < TRAVERSAL_STACK_SIZE){
                blas_stack[stack_pointer] = far_index;
                stack_pointer++;
            }
            if(near_t < (*best).t){
                node_index = near_index;
                continue;
            }
        }
        if(stack_pointer == 0u){ return found; }
        stack_pointer--;
        node_index = blas_stack[stack_pointer];
    }
    return found;
}

// one instance-level traversal for the scene TLAS and every overlay TLAS: every inlined copy of
// the nested BLAS loop costs kernel time (a separate overlay copy cost about a fifth of it).
// Overlay TLAS leaves index their instance ids in FRAME_INPUTS
fn traverse_tlas(node_base: u32, primitive_base: u32, overlay: bool, origin: vec3<f32>, direction: vec3<f32>, inv_direction: vec3<f32>, t_min: f32, best: ptr<function, Hit>, any_hit: bool) -> bool{
    if(intersect_aabb_distance(bvh_nodes[node_base], origin, inv_direction, t_min, (*best).t) >= (*best).t){ return false; }
    var stack_pointer = 0u;
    var node_index = 0u;
    loop{
        let node = bvh_nodes[node_base + node_index];
        let left_or_first = bvh_left_or_first(node);
        let count = bvh_count(node);
        if(count > 0u){
            for(var i = 0u; i < count; i++){
                var instance_index: u32;
                if(overlay){
                    instance_index = frame_u32(primitive_base + left_or_first + i);
                }
                else{
                    instance_index = geometry_u32(primitive_base + left_or_first + i);
                }
                if(intersect_blas(instance_index, origin, direction, inv_direction, t_min, best, any_hit) && any_hit){ return true; }
            }
        }
        else{
            let t_left = intersect_aabb_distance(bvh_nodes[node_base + left_or_first], origin, inv_direction, t_min, (*best).t);
            let t_right = intersect_aabb_distance(bvh_nodes[node_base + left_or_first + 1u], origin, inv_direction, t_min, (*best).t);
            var near_index = left_or_first;
            var near_t = t_left;
            var far_index = left_or_first + 1u;
            var far_t = t_right;
            if(t_right < t_left){
                near_index = left_or_first + 1u;
                near_t = t_right;
                far_index = left_or_first;
                far_t = t_left;
            }
            if(far_t < (*best).t && stack_pointer < TRAVERSAL_STACK_SIZE){
                tlas_stack[stack_pointer] = far_index;
                stack_pointer++;
            }
            if(near_t < (*best).t){
                node_index = near_index;
                continue;
            }
        }
        if(stack_pointer == 0u){ return false; }
        stack_pointer--;
        node_index = tlas_stack[stack_pointer];
    }
    return false;
}

// min-t composition of the shared world and the overlays attached to this camera; at
// fc_overlay_count == 0 the loop specializes away and this is a plain intersect
// the overlay region selects shutter close (0), a dynamic motion sample, or the flow previous instant
fn trace_composed_region(region: u32, camera: i32, origin: vec3<f32>, direction: vec3<f32>, t_min: f32, t_max: f32, any_hit: bool) -> Hit{
    var best: Hit;
    best.t = t_max;
    best.u = 0.0;
    best.v = 0.0;
    best.triangle = 0u;
    best.instance = 0u;
    best.valid = false;
    let inv_direction = safe_inverse(direction);
    // k = -1 is the scene TLAS, k >= 0 the k-th overlay attached to this camera
    for(var k = -1; k < fc_overlay_count; k++){
        var node_base = params.tlas_node_offset;
        var primitive_base = params.tlas_primitive_offset;
        if(k < 0){
            if(params.tlas_node_count == 0u){ continue; }
        }
        else{
            let overlay = frame_u32(params.attachments_offset + u32(camera) * u32(fc_overlay_count) + u32(k));
            if(overlay == ABSENT){ continue; }
            let slot = region * fc_num_overlays + overlay;
            if(frame_u32(params.overlay_meta_offset + slot * 2u + 1u) == 0u){ continue; }
            node_base = params.overlay_node_offset + slot * 2u * fc_overlay_capacity;
            primitive_base = params.overlay_primitives_offset + slot * fc_overlay_capacity;
        }
        if(traverse_tlas(node_base, primitive_base, k >= 0, origin, direction, inv_direction, t_min, &best, any_hit) && any_hit){ return best; }
    }
    return best;
}

fn trace_closest_composed_region(region: u32, camera: i32, origin: vec3<f32>, direction: vec3<f32>, t_min: f32, t_max: f32) -> Hit{
    return trace_composed_region(region, camera, origin, direction, t_min, t_max, false);
}

fn trace_any_composed_region(region: u32, camera: i32, origin: vec3<f32>, direction: vec3<f32>, t_min: f32, t_max: f32) -> bool{
    return trace_composed_region(region, camera, origin, direction, t_min, t_max, true).valid;
}

struct PixelLaunchContext{
    cam_idx: i32,
    local_x: i32,
    local_y: i32,
    fb_offset: i32,
    valid: bool,
}

fn pixel_launch_context(pixel_id: vec2<u32>) -> PixelLaunchContext{
    var ctx: PixelLaunchContext;
    let tile_col = i32(pixel_id.x / params.cam_width);
    let tile_row = i32(pixel_id.y / params.cam_height);
    ctx.cam_idx = tile_row * i32(params.grid_cols) + tile_col;
    ctx.local_x = i32(pixel_id.x) - tile_col * i32(params.cam_width);
    ctx.local_y = i32(pixel_id.y) - tile_row * i32(params.cam_height);
    ctx.fb_offset = ctx.cam_idx * i32(params.cam_width) * i32(params.cam_height) + ctx.local_y * i32(params.cam_width) + ctx.local_x;
    ctx.valid = pixel_id.x < params.fb_width && pixel_id.y < params.fb_height && ctx.cam_idx < i32(params.num_cameras);
    return ctx;
}

fn make_8bit(f: f32) -> u32{
    return u32(min(255, max(0, i32(f * 256.0))));
}

fn make_rgba(color: vec3<f32>) -> u32{
    return (make_8bit(color.x) << 0u) | (make_8bit(color.y) << 8u) | (make_8bit(color.z) << 16u) | (0xFFu << 24u);
}

fn linear_to_srgb(x: f32) -> f32{
    if(x <= 0.0031308){
        return 12.92 * x;
    }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

fn make_srgb_rgba_from_linear(color: vec3<f32>) -> u32{
    return make_rgba(vec3<f32>(
        linear_to_srgb(clamp(color.x, 0.0, 1.0)),
        linear_to_srgb(clamp(color.y, 0.0, 1.0)),
        linear_to_srgb(clamp(color.z, 0.0, 1.0))));
}

fn make_linear_rgba_from_linear(color: vec3<f32>) -> u32{
    return make_rgba(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)));
}

fn miss_color(pixel_id: vec2<u32>) -> vec3<f32>{
    if(fc_checker_background){
        let checker_pattern = (i32(pixel_id.x) / 8) ^ (i32(pixel_id.y) / 8);
        if((checker_pattern & 1) != 0){
            return to_vec3(params.miss_color_1);
        }
        return to_vec3(params.miss_color_0);
    }
    return to_vec3(params.miss_color_0);
}

fn trace_shadow_occluded(region: u32, camera: i32, pos: vec3<f32>, direction: vec3<f32>, max_dist: f32) -> bool{
    return trace_any_composed_region(region, camera, pos, direction, 1e-3, max_dist);
}

// host-precomputed 256-entry table; unpack4x8unorm yields exact k/255, so the index is exact
fn srgb_texel_to_linear(x: f32) -> f32{
    return geometry_f32(params.srgb_lut_offset + u32(round(x * 255.0)));
}

// GPU-style normalized-coordinate bilinear sampling with repeat wrap; sRGB decode happens per
// texel before filtering — mirrors backends/generic/operations_generic.h::sample_texture
fn sample_texture(texture: TextureRef, u: f32, v: f32, srgb: bool) -> vec4<f32>{
    let width = i32(texture.width);
    let height = i32(texture.height);
    let x = u * f32(width) - 0.5;
    let y = v * f32(height) - 0.5;
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    let x0_wrapped = ((x0 % width) + width) % width;
    let x1_wrapped = (((x0 + 1) % width) + width) % width;
    let y0_wrapped = ((y0 % height) + height) % height;
    let y1_wrapped = (((y0 + 1) % height) + height) % height;
    var texel_00 = unpack4x8unorm(texture_data[texture.offset + u32(y0_wrapped * width + x0_wrapped)]);
    var texel_10 = unpack4x8unorm(texture_data[texture.offset + u32(y0_wrapped * width + x1_wrapped)]);
    var texel_01 = unpack4x8unorm(texture_data[texture.offset + u32(y1_wrapped * width + x0_wrapped)]);
    var texel_11 = unpack4x8unorm(texture_data[texture.offset + u32(y1_wrapped * width + x1_wrapped)]);
    if(srgb){
        texel_00 = vec4<f32>(srgb_texel_to_linear(texel_00.x), srgb_texel_to_linear(texel_00.y), srgb_texel_to_linear(texel_00.z), texel_00.w);
        texel_10 = vec4<f32>(srgb_texel_to_linear(texel_10.x), srgb_texel_to_linear(texel_10.y), srgb_texel_to_linear(texel_10.z), texel_10.w);
        texel_01 = vec4<f32>(srgb_texel_to_linear(texel_01.x), srgb_texel_to_linear(texel_01.y), srgb_texel_to_linear(texel_01.z), texel_01.w);
        texel_11 = vec4<f32>(srgb_texel_to_linear(texel_11.x), srgb_texel_to_linear(texel_11.y), srgb_texel_to_linear(texel_11.z), texel_11.w);
    }
    let top = texel_00 + fx * (texel_10 - texel_00);
    let bottom = texel_01 + fx * (texel_11 - texel_01);
    return top + fy * (bottom - top);
}

struct SecondaryRequest{
    want_reflection: bool,
    reflect_dir: vec3<f32>,
    reflection_weight: f32, // basic: fresnel factor, pbr: blend weight
    want_transparency: bool,
    alpha: f32,
    alpha_mode: i32,
    hit_point: vec3<f32>,
    ray_dir: vec3<f32>,
}

fn shade_basic_local(hit: Hit, ray_origin: vec3<f32>, ray_dir: vec3<f32>, request: ptr<function, SecondaryRequest>) -> vec3<f32>{
    let mesh = fetch_triangle_mesh(hit.triangle);
    let rec = load_mesh_record(mesh);
    (*request).want_reflection = false;
    (*request).want_transparency = false;

    var base_color = to_vec3(rec.color);
    var normal_geometric = vec3<f32>(0.0, 0.0, 1.0);

    if(fc_load_textures || fc_normal_shading || fc_metallic_reflections){
        let index = fetch_index(mesh, fetch_triangle_local(hit.triangle));

        if(fc_normal_shading || fc_metallic_reflections){
            var vertex_a = fetch_vertex(mesh, index.x);
            var vertex_b = fetch_vertex(mesh, index.y);
            var vertex_c = fetch_vertex(mesh, index.z);
            if(instance_data[hit.instance].identity == 0u){
                let object_to_world = instance_data[hit.instance].object_to_world;
                vertex_a = transform_point(object_to_world, vertex_a);
                vertex_b = transform_point(object_to_world, vertex_b);
                vertex_c = transform_point(object_to_world, vertex_c);
            }
            normal_geometric = normalize(cross(vertex_b - vertex_a, vertex_c - vertex_a));
        }

        if(fc_load_textures){
            if(rec.texture.offset != ABSENT && rec.tex_coord_offset != ABSENT){
                let tc = (1.0 - hit.u - hit.v) * fetch_tex_coord(mesh, index.x)
                    + hit.u * fetch_tex_coord(mesh, index.y)
                    + hit.v * fetch_tex_coord(mesh, index.z);
                let tex_color = sample_texture(rec.texture, tc.x, tc.y, true);
                base_color = tex_color.xyz * to_vec3(rec.color);
            }
        }
    }

    if(rec.unlit != 0){
        return base_color;
    }

    var direct = base_color;
    if(fc_normal_shading){
        direct = (0.2 + 0.8 * abs(dot(ray_dir, normal_geometric))) * base_color;
    }

    if(fc_metallic_reflections){
        if(rec.metallic > 0.0){
            let hit_point = ray_origin + ray_dir * hit.t;
            let n = select(normal_geometric, -normal_geometric, dot(ray_dir, normal_geometric) > 0.0);
            let cos_theta = abs(dot(ray_dir, n));
            (*request).want_reflection = true;
            (*request).reflect_dir = ray_dir - 2.0 * dot(ray_dir, n) * n;
            (*request).reflection_weight = rec.metallic * (0.04 + 0.96 * pow(1.0 - cos_theta, 5.0));
            (*request).hit_point = hit_point;
            (*request).ray_dir = ray_dir;
        }
    }
    return direct;
}

fn shade_pbr_local(region: u32, hit: Hit, ray_origin: vec3<f32>, ray_dir: vec3<f32>, camera: i32, request: ptr<function, SecondaryRequest>) -> vec3<f32>{
    let mesh = fetch_triangle_mesh(hit.triangle);
    let rec = load_mesh_record(mesh);
    (*request).want_reflection = false;
    (*request).want_transparency = false;

    let instance_identity = instance_data[hit.instance].identity != 0u;
    let index = fetch_index(mesh, fetch_triangle_local(hit.triangle));
    var vertex_a = fetch_vertex(mesh, index.x);
    var vertex_b = fetch_vertex(mesh, index.y);
    var vertex_c = fetch_vertex(mesh, index.z);
    if(!instance_identity){
        let object_to_world = instance_data[hit.instance].object_to_world;
        vertex_a = transform_point(object_to_world, vertex_a);
        vertex_b = transform_point(object_to_world, vertex_b);
        vertex_c = transform_point(object_to_world, vertex_c);
    }
    let w0 = 1.0 - hit.u - hit.v;

    let edge1 = vertex_b - vertex_a;
    let edge2 = vertex_c - vertex_a;
    let normal_geometric = normalize(cross(edge1, edge2));

    var N: vec3<f32>;
    if(rec.normal_offset != ABSENT){
        var normal_interpolated = w0 * fetch_normal(mesh, index.x) + hit.u * fetch_normal(mesh, index.y) + hit.v * fetch_normal(mesh, index.z);
        if(!instance_identity){
            normal_interpolated = transform_normal(instance_data[hit.instance].world_to_object, normal_interpolated);
        }
        N = normalize(normal_interpolated);
    }
    else{
        N = normal_geometric;
    }

    if(dot(ray_dir, N) > 0.0){
        N = -N;
    }

    let has_tex_coord = rec.tex_coord_offset != ABSENT;
    var tc = vec2<f32>(0.0);
    if(has_tex_coord){
        tc = w0 * fetch_tex_coord(mesh, index.x) + hit.u * fetch_tex_coord(mesh, index.y) + hit.v * fetch_tex_coord(mesh, index.z);
    }

    var base_color = to_vec3(rec.color);
    var alpha = rec.opacity;
    if(rec.texture.offset != ABSENT && has_tex_coord){
        let tex_color = sample_texture(rec.texture, tc.x, tc.y, true);
        base_color = tex_color.xyz * to_vec3(rec.color);
        alpha *= tex_color.w;
    }

    let hit_point = ray_origin + ray_dir * hit.t;

    var color = base_color;
    if(rec.unlit == 0){
        var metallic = rec.metallic;
        var roughness = rec.roughness;
        if(rec.metallic_roughness_map.offset != ABSENT && has_tex_coord){
            let mr_sample = sample_texture(rec.metallic_roughness_map, tc.x, tc.y, false);
            roughness = mr_sample.y * rec.roughness;
            metallic = mr_sample.z * rec.metallic;
        }

        if(rec.normal_map.offset != ABSENT && has_tex_coord){
            let tc0 = fetch_tex_coord(mesh, index.x);
            let tc1 = fetch_tex_coord(mesh, index.y);
            let tc2 = fetch_tex_coord(mesh, index.z);
            let duv1 = tc1 - tc0;
            let duv2 = tc2 - tc0;
            let det = duv1.x * duv2.y - duv2.x * duv1.y;
            if(abs(det) > 1e-8){
                let inv_det = 1.0 / det;
                var T = normalize(inv_det * (duv2.y * edge1 - duv1.y * edge2));
                T = normalize(T - dot(T, N) * N);
                let B = cross(N, T);
                let nm_sample = sample_texture(rec.normal_map, w0 * tc0.x + hit.u * tc1.x + hit.v * tc2.x, w0 * tc0.y + hit.u * tc1.y + hit.v * tc2.y, false);
                let n_tangent = vec3<f32>(nm_sample.x * 2.0 - 1.0, -(nm_sample.y * 2.0 - 1.0), nm_sample.z * 2.0 - 1.0);
                N = normalize(T * n_tangent.x + B * n_tangent.y + N * n_tangent.z);
            }
        }

        roughness = max(roughness, 0.04);
        let roughness_alpha = roughness * roughness;
        let alpha2 = roughness_alpha * roughness_alpha;
        let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;

        let V = -ray_dir;
        let NdotV = max(dot(N, V), 1e-4);
        let F0 = vec3<f32>(0.04) * (1.0 - metallic) + base_color * metallic;

        var Lo = vec3<f32>(0.0);
        for(var li = 0u; li < params.num_scene_lights; li++){
            let light = load_scene_light(li);
            let Lc = to_vec3(light.color);
            var L: vec3<f32>;
            var attenuation = 1.0;
            var light_distance = 1e20;

            if(light.light_type == 0){
                L = to_vec3(light.direction);
            }
            else{
                let to_light = to_vec3(light.position) - hit_point;
                let dist = length(to_light);
                light_distance = max(dist - 1e-3, 0.0);
                L = to_light * (1.0 / max(dist, 1e-6));
                attenuation = 1.0 / (light.attenuation_constant + light.attenuation_linear * dist + light.attenuation_quadratic * dist * dist);
                if(light.light_type == 2){
                    let spot_dir = to_vec3(light.direction);
                    let cos_angle = dot(-L, spot_dir);
                    let denom = light.cos_inner_cone - light.cos_outer_cone;
                    let spot_t = (cos_angle - light.cos_outer_cone) / select(1e-6, denom, abs(denom) > 1e-6);
                    attenuation *= max(min(spot_t, 1.0), 0.0);
                }
            }

            let NdotL = max(dot(N, L), 0.0);
            if(NdotL <= 0.0){
                continue;
            }
            if(fc_punctual_light_shadows){
                if(light.light_type != 0 && trace_shadow_occluded(region, camera, hit_point + N * 2e-3, L, light_distance)){
                    continue;
                }
            }

            let H = normalize(V + L);
            let NdotH = max(dot(N, H), 0.0);
            let VdotH = max(dot(V, H), 0.0);

            let denom_D = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
            let D = alpha2 / (PI * denom_D * denom_D);

            let G1_V = NdotV / (NdotV * (1.0 - k) + k);
            let G1_L = NdotL / (NdotL * (1.0 - k) + k);
            let G = G1_V * G1_L;

            let pow5 = pow(1.0 - VdotH, 5.0);
            let F = F0 + (vec3<f32>(1.0) - F0) * pow5;

            let specular = D * G * F * (1.0 / (4.0 * NdotV * NdotL + 1e-4));
            let kd = (vec3<f32>(1.0) - F) * (1.0 - metallic);
            let diffuse = kd * base_color * (1.0 / PI);

            Lo = Lo + (diffuse + specular) * Lc * (attenuation * NdotL);
        }

        var occlusion = 1.0;
        if(rec.occlusion_map.offset != ABSENT && has_tex_coord){
            let ao_sample = sample_texture(rec.occlusion_map, tc.x, tc.y, false);
            occlusion = ao_sample.x;
        }

        var emissive_color: vec3<f32>;
        if(rec.emissive_map.offset != ABSENT && has_tex_coord){
            let em_sample = sample_texture(rec.emissive_map, tc.x, tc.y, true);
            emissive_color = em_sample.xyz * to_vec3(rec.emissive);
        }
        else{
            emissive_color = to_vec3(rec.emissive);
        }

        let ambient = to_vec3(params.ambient_color) * base_color * ((1.0 - metallic) * occlusion);
        color = ambient + Lo + emissive_color;

        if(metallic > 0.1){
            let fresnel_refl = F0.x + (1.0 - F0.x) * pow(1.0 - max(dot(V, N), 0.0), 5.0);
            (*request).want_reflection = true;
            (*request).reflect_dir = ray_dir - 2.0 * dot(ray_dir, N) * N;
            (*request).reflection_weight = fresnel_refl * (1.0 - roughness);
        }

    }

    let transparent = (rec.alpha_mode == 2 && alpha < 0.99) || (rec.alpha_mode == 1 && alpha < rec.alpha_cutoff);
    if(transparent){
        (*request).want_transparency = true;
    }
    (*request).alpha = alpha;
    (*request).alpha_mode = rec.alpha_mode;
    (*request).hit_point = hit_point;
    (*request).ray_dir = ray_dir;

    return color;
}

fn trace_leaf(region: u32, camera: i32, origin: vec3<f32>, direction: vec3<f32>, t_min: f32, pixel_id: vec2<u32>) -> vec3<f32>{
    let hit = trace_closest_composed_region(region, camera, origin, direction, t_min, 1e20);
    if(!hit.valid){
        return miss_color(pixel_id);
    }
    var request: SecondaryRequest;
    if(fc_pbr_shading){
        return shade_pbr_local(region, hit, origin, direction, camera, &request);
    }
    return shade_basic_local(hit, origin, direction, &request);
}

fn shade_primary_hit(region: u32, hit: Hit, camera: i32, origin: vec3<f32>, direction: vec3<f32>, pixel_id: vec2<u32>) -> vec3<f32>{
    if(!hit.valid){
        return miss_color(pixel_id);
    }
    var request: SecondaryRequest;
    if(fc_pbr_shading){
        var color = shade_pbr_local(region, hit, origin, direction, camera, &request);
        if(request.want_reflection){
            let reflected_color = trace_leaf(region, camera, request.hit_point, request.reflect_dir, 1e-3, pixel_id);
            color = color * (1.0 - request.reflection_weight) + reflected_color * request.reflection_weight;
        }
        if(request.want_transparency){
            let behind_color = trace_leaf(region, camera, request.hit_point, request.ray_dir, 1e-5, pixel_id);
            if(request.alpha_mode == 1){
                color = behind_color;
            }
            else{
                color = color * request.alpha + behind_color * (1.0 - request.alpha);
            }
        }
        return color;
    }
    let direct = shade_basic_local(hit, origin, direction, &request);
    if(fc_metallic_reflections && request.want_reflection){
        let reflected_color = trace_leaf(region, camera, request.hit_point, request.reflect_dir, 1e-3, pixel_id);
        let fresnel = request.reflection_weight;
        return direct * ((1.0 - fresnel) + fresnel * reflected_color);
    }
    return direct;
}

fn trace_primary(region: u32, camera: i32, origin: vec3<f32>, direction: vec3<f32>, t_min: f32, t_max: f32, pixel_id: vec2<u32>) -> vec3<f32>{
    let hit = trace_closest_composed_region(region, camera, origin, direction, t_min, t_max);
    return shade_primary_hit(region, hit, camera, origin, direction, pixel_id);
}

struct ShutterCamera{
    pos: vec3<f32>,
    dir_00: vec3<f32>,
    dir_du: vec3<f32>,
    dir_dv: vec3<f32>,
}

fn shutter_camera(cam_idx: i32, motion_i: i32) -> ShutterCamera{
    var result: ShutterCamera;
    if(fc_motion_blur){
        let cam_open = camera_open(cam_idx);
        let cam_close = camera_close(cam_idx);
        let shutter_t = (f32(motion_i) + 0.5) * (1.0 / f32(fc_motion_samples));
        result.pos = mix(camera_pos(cam_open), camera_pos(cam_close), shutter_t);
        result.dir_00 = mix(camera_dir_00(cam_open), camera_dir_00(cam_close), shutter_t);
        result.dir_du = mix(camera_dir_du(cam_open), camera_dir_du(cam_close), shutter_t);
        result.dir_dv = mix(camera_dir_dv(cam_open), camera_dir_dv(cam_close), shutter_t);
    }
    else{
        let cam = camera_close(cam_idx);
        result.pos = camera_pos(cam);
        result.dir_00 = camera_dir_00(cam);
        result.dir_du = camera_dir_du(cam);
        result.dir_dv = camera_dir_dv(cam);
    }
    return result;
}

@compute @workgroup_size(8, 4, 1)
fn main_rgb(@builtin(global_invocation_id) global_id: vec3<u32>){
    let pixel_id = global_id.xy;
    let ctx = pixel_launch_context(pixel_id);
    if(!ctx.valid){
        return;
    }

    var accumulated = vec3<f32>(0.0);
    let inv_aa_grid = 1.0 / f32(fc_aa_grid);
    for(var motion_i = 0; motion_i < fc_motion_samples; motion_i++){
        let region = select(0u, 1u + u32(motion_i), fc_dynamic_motion_blur);
        let cam = shutter_camera(ctx.cam_idx, motion_i);
        var sample_color = vec3<f32>(0.0);
        for(var aa_y = 0; aa_y < fc_aa_grid; aa_y++){
            for(var aa_x = 0; aa_x < fc_aa_grid; aa_x++){
                let screen = (vec2<f32>(f32(ctx.local_x), f32(ctx.local_y)) + vec2<f32>((f32(aa_x) + 0.5) * inv_aa_grid, (f32(aa_y) + 0.5) * inv_aa_grid)) / vec2<f32>(f32(params.cam_width), f32(params.cam_height));
                let direction = normalize(cam.dir_00 + screen.x * cam.dir_du + screen.y * cam.dir_dv);
                var color: vec3<f32>;
                if(fc_fuse_auxiliary){
                    let hit = trace_closest_composed_region(region, ctx.cam_idx, cam.pos, direction, 0.0, 1e30);
                    write_auxiliary(ctx, camera_close(ctx.cam_idx), direction, hit);
                    color = shade_primary_hit(region, hit, ctx.cam_idx, cam.pos, direction, pixel_id);
                }
                else{
                    color = trace_primary(region, ctx.cam_idx, cam.pos, direction, 0.0, 1e30, pixel_id);
                }
                if(fc_dynamic_motion_blur){
                    sample_color += color;
                }
                else{
                    accumulated += color;
                }
            }
        }
        if(fc_dynamic_motion_blur){
            accumulated += sample_color * (1.0 / f32(fc_aa_grid * fc_aa_grid));
        }
    }
    let samples = select(fc_motion_samples * fc_aa_grid * fc_aa_grid, fc_motion_samples, fc_dynamic_motion_blur);
    let color = accumulated * (1.0 / f32(samples));
    if(output_enabled(OUTPUT_RGB_FLOAT32)){
        // the observation is the frame-buffer color before 8-bit quantization
        var obs_color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
        if(fc_srgb_output){
            obs_color = vec3<f32>(linear_to_srgb(obs_color.x), linear_to_srgb(obs_color.y), linear_to_srgb(obs_color.z));
        }
        out_set_f32(params.output_offsets[OUTPUT_RGB_FLOAT32] + u32(ctx.fb_offset * 3 + 0), obs_color.x);
        out_set_f32(params.output_offsets[OUTPUT_RGB_FLOAT32] + u32(ctx.fb_offset * 3 + 1), obs_color.y);
        out_set_f32(params.output_offsets[OUTPUT_RGB_FLOAT32] + u32(ctx.fb_offset * 3 + 2), obs_color.z);
    }
    if(output_enabled(OUTPUT_RGB_RGBA8)){
        if(fc_srgb_output){
            outputs[params.output_offsets[OUTPUT_RGB_RGBA8] + u32(ctx.fb_offset)] = make_srgb_rgba_from_linear(color);
        }
        else{
            outputs[params.output_offsets[OUTPUT_RGB_RGBA8] + u32(ctx.fb_offset)] = make_linear_rgba_from_linear(color);
        }
    }
}

@compute @workgroup_size(8, 4, 1)
fn main_collision(@builtin(global_invocation_id) global_id: vec3<u32>){
    let cam_idx = i32(global_id.x);
    let probe_idx = i32(global_id.y);

    if(cam_idx >= i32(params.num_cameras) || probe_idx >= i32(params.num_probes)){
        return;
    }

    let cam = camera_close(cam_idx);

    let forward = normalize(camera_dir_00(cam) + 0.5 * camera_dir_du(cam) + 0.5 * camera_dir_dv(cam));
    var dir: vec3<f32>;
    if(probe_idx == 0){
        dir = forward;
    }
    else{
        let local = vec3<f32>(
            frame_f32(params.probe_directions_offset + u32(3 * probe_idx)),
            frame_f32(params.probe_directions_offset + u32(3 * probe_idx + 1)),
            frame_f32(params.probe_directions_offset + u32(3 * probe_idx + 2)));
        let left = -normalize(camera_dir_du(cam));
        let up = -normalize(camera_dir_dv(cam));
        dir = normalize(local.x * forward + local.y * left + local.z * up);
    }

    let hit = trace_closest_composed_region(0u, cam_idx, camera_pos(cam), dir, 1e-3, params.max_dist);

    // two OUTPUTS words per probe: {distance: f32, hit: u32}
    let result_base = params.output_offsets[OUTPUT_PROBES] + u32(cam_idx * i32(params.num_probes) + probe_idx) * 2u;
    if(!hit.valid){
        outputs[result_base] = 0x7f800000u;
        outputs[result_base + 1u] = 0u;
    }
    else{
        out_set_f32(result_base, hit.t);
        outputs[result_base + 1u] = 1u;
    }
}

fn apply_flow_delta(base_in: u32, p: vec3<f32>) -> vec3<f32>{
    let base = params.flow_deltas_offset + base_in;
    return vec3<f32>(
        frame_f32(base + 0u)*p.x + frame_f32(base + 1u)*p.y + frame_f32(base + 2u)*p.z + frame_f32(base + 3u),
        frame_f32(base + 4u)*p.x + frame_f32(base + 5u)*p.y + frame_f32(base + 6u)*p.z + frame_f32(base + 7u),
        frame_f32(base + 8u)*p.x + frame_f32(base + 9u)*p.y + frame_f32(base + 10u)*p.z + frame_f32(base + 11u));
}

// world point → screen fraction through the linear camera model: solves
// dir_00 + screen_x·dir_du + screen_y·dir_dv = lambda·(point − pos) by Cramer's rule;
// returns w = 0 for a degenerate basis or a point at/behind the camera (lambda <= 0)
fn project_camera(cam: Camera, point: vec3<f32>) -> vec3<f32>{
    let direction = point - camera_pos(cam);
    let dir_du = camera_dir_du(cam);
    let dir_dv = camera_dir_dv(cam);
    let negative_direction = -direction;
    let cross_dv_negative_direction = cross(dir_dv, negative_direction);
    let det = dot(dir_du, cross_dv_negative_direction);
    if(det > -1e-12 && det < 1e-12){
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let inv_det = 1.0 / det;
    let b = -camera_dir_00(cam);
    let screen_x = dot(b, cross_dv_negative_direction) * inv_det;
    let screen_y = dot(dir_du, cross(b, negative_direction)) * inv_det;
    let lambda = dot(dir_du, cross(dir_dv, b)) * inv_det;
    return vec3<f32>(screen_x, screen_y, select(0.0, 1.0, lambda > 0.0));
}

// FlowStatus codes (rl_tools/rendering/raytracing/types.h)
const FLOW_STATUS_VALID: u32 = 0u;
const FLOW_STATUS_MISS: u32 = 1u;
const FLOW_STATUS_BEHIND: u32 = 2u;
const FLOW_STATUS_OUT_OF_FRAME: u32 = 3u;
const FLOW_STATUS_OCCLUDED: u32 = 4u;
const FLOW_VISIBILITY_EPSILON: f32 = 1e-4;

// two-instant flow, one direction per entry point (mirror of the generic render_flow_frame):
// single-sample from the source pixel centre against the source instant's overlay region; the
// hit point is carried to the target instant by its slot's delta, projected through the target
// camera, and tested for visibility from the target camera against the target instant with the
// relative margin FLOW_VISIBILITY_EPSILON. Status priority MISS, BEHIND, OUT_OF_FRAME, OCCLUDED,
// VALID; the vector is zero for MISS and BEHIND and the geometric flow otherwise
fn write_flow(source_previous: bool, ctx: PixelLaunchContext, cam: Camera, direction: vec3<f32>, hit: Hit){
    let previous_region = 1u + select(0u, u32(fc_motion_samples), fc_dynamic_motion_blur);
    let source_region = select(0u, previous_region, source_previous);
    let target_region = select(previous_region, 0u, source_previous);
    var cam_target: Camera;
    if(source_previous){
        cam_target = camera_close(ctx.cam_idx);
    }
    else{
        cam_target = camera_previous(ctx.cam_idx);
    }
    let source_first_overlay_instance = params.first_overlay_instance + params.num_overlay_slots * source_region;
    let delta_slab = select(0u, params.num_overlay_slots * 12u, source_previous);
    var flow = vec2<f32>(0.0);
    var status = FLOW_STATUS_MISS;
    if(hit.valid){
        var point = camera_pos(cam) + direction * hit.t;
        if(fc_overlay_count > 0 && hit.instance >= source_first_overlay_instance){
            point = apply_flow_delta(delta_slab + (hit.instance - source_first_overlay_instance) * 12u, point);
        }
        let projected = project_camera(cam_target, point);
        if(projected.z < 0.5){
            status = FLOW_STATUS_BEHIND;
        }
        else{
            flow = vec2<f32>(projected.x * f32(params.cam_width) - (f32(ctx.local_x) + 0.5),
                             projected.y * f32(params.cam_height) - (f32(ctx.local_y) + 0.5));
            if(projected.x < 0.0 || projected.x >= 1.0 || projected.y < 0.0 || projected.y >= 1.0){
                status = FLOW_STATUS_OUT_OF_FRAME;
            }
            else if((fc_output_mask & (7u << select(OUTPUT_FLOW_RGBA8, OUTPUT_FLOW_FORWARD_RGBA8, source_previous))) != 0u){
                let to_point = point - camera_pos(cam_target);
                let len = length(to_point);
                let visibility_direction = to_point * (1.0 / len);
                let occluded = trace_any_composed_region(target_region, ctx.cam_idx, camera_pos(cam_target), visibility_direction, 0.0, len * (1.0 - FLOW_VISIBILITY_EPSILON));
                status = select(FLOW_STATUS_VALID, FLOW_STATUS_OCCLUDED, occluded);
            }
        }
    }
    let flow_slot = select(OUTPUT_FLOW_FLOAT32, OUTPUT_FLOW_FORWARD_FLOAT32, source_previous);
    let color_slot = select(OUTPUT_FLOW_RGBA8, OUTPUT_FLOW_FORWARD_RGBA8, source_previous);
    let status_slot = select(OUTPUT_FLOW_STATUS_UINT32, OUTPUT_FLOW_FORWARD_STATUS_UINT32, source_previous);
    let status_color_slot = select(OUTPUT_FLOW_STATUS_RGBA8, OUTPUT_FLOW_FORWARD_STATUS_RGBA8, source_previous);
    if(output_enabled(flow_slot)){
        out_set_f32(params.output_offsets[flow_slot] + u32(ctx.fb_offset * 2), flow.x);
        out_set_f32(params.output_offsets[flow_slot] + u32(ctx.fb_offset * 2 + 1), flow.y);
    }
    if(output_enabled(color_slot)){
        var color = flow_status_to_rgba(status);
        if(status == FLOW_STATUS_VALID){
            color = flow_to_rgba(flow.x, flow.y, params.output_scales[color_slot]);
        }
        outputs[params.output_offsets[color_slot] + u32(ctx.fb_offset)] = color;
    }
    if(output_enabled(status_slot)){
        outputs[params.output_offsets[status_slot] + u32(ctx.fb_offset)] = status;
    }
    if(output_enabled(status_color_slot)){
        outputs[params.output_offsets[status_color_slot] + u32(ctx.fb_offset)] = flow_status_to_rgba(status);
    }
}

fn write_auxiliary(ctx: PixelLaunchContext, cam: Camera, direction: vec3<f32>, hit: Hit){
    if(output_enabled(OUTPUT_DEPTH_FLOAT32) || output_enabled(OUTPUT_DEPTH_RGBA8)){
        let normal = normalize(cross(camera_dir_du(cam), camera_dir_dv(cam)));
        var depth = 0.0;
        if(hit.valid){
            depth = hit.t * abs(dot(direction, normal));
        }
        if(output_enabled(OUTPUT_DEPTH_FLOAT32)){
            outputs[params.output_offsets[OUTPUT_DEPTH_FLOAT32] + u32(ctx.fb_offset)] = select(0x7f800000u, bitcast<u32>(depth), hit.valid);
        }
        if(output_enabled(OUTPUT_DEPTH_RGBA8)){
            var color = 0xFFFFFFFFu;
            if(hit.valid){
                color = depth_to_rgba(depth, params.output_scales[OUTPUT_DEPTH_RGBA8]);
            }
            outputs[params.output_offsets[OUTPUT_DEPTH_RGBA8] + u32(ctx.fb_offset)] = color;
        }
    }
    if(output_enabled(OUTPUT_SEGMENTATION_UINT32) || output_enabled(OUTPUT_SEGMENTATION_RGBA8)){
        let value = select(ABSENT, hit.instance, hit.valid);
        if(output_enabled(OUTPUT_SEGMENTATION_UINT32)){
            outputs[params.output_offsets[OUTPUT_SEGMENTATION_UINT32] + u32(ctx.fb_offset)] = value;
        }
        if(output_enabled(OUTPUT_SEGMENTATION_RGBA8)){
            outputs[params.output_offsets[OUTPUT_SEGMENTATION_RGBA8] + u32(ctx.fb_offset)] = segmentation_id_to_rgba(value);
        }
    }
    if(output_enabled(OUTPUT_SEMANTIC_UINT32) || output_enabled(OUTPUT_SEMANTIC_RGBA8)){
        var value = ABSENT;
        if(hit.valid){
            value = geometry_u32(params.instance_classes_offset + hit.instance);
        }
        if(output_enabled(OUTPUT_SEMANTIC_UINT32)){
            outputs[params.output_offsets[OUTPUT_SEMANTIC_UINT32] + u32(ctx.fb_offset)] = value;
        }
        if(output_enabled(OUTPUT_SEMANTIC_RGBA8)){
            outputs[params.output_offsets[OUTPUT_SEMANTIC_RGBA8] + u32(ctx.fb_offset)] = segmentation_id_to_rgba(value);
        }
    }
    if(output_enabled(OUTPUT_NORMALS_FLOAT32) || output_enabled(OUTPUT_NORMALS_RGBA8)){
        var normal = vec3<f32>(0.0);
        if(hit.valid){
            var vertices = triangle_vertices(hit.triangle);
            if(instance_data[hit.instance].identity == 0u){
                let object_to_world = instance_data[hit.instance].object_to_world;
                vertices[0] = transform_point(object_to_world, vertices[0]);
                vertices[1] = transform_point(object_to_world, vertices[1]);
                vertices[2] = transform_point(object_to_world, vertices[2]);
            }
            normal = normalize(cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
            if(dot(direction, normal) > 0.0){
                normal = -normal;
            }
        }
        if(output_enabled(OUTPUT_NORMALS_FLOAT32)){
            out_set_f32(params.output_offsets[OUTPUT_NORMALS_FLOAT32] + u32(ctx.fb_offset * 3), normal.x);
            out_set_f32(params.output_offsets[OUTPUT_NORMALS_FLOAT32] + u32(ctx.fb_offset * 3 + 1), normal.y);
            out_set_f32(params.output_offsets[OUTPUT_NORMALS_FLOAT32] + u32(ctx.fb_offset * 3 + 2), normal.z);
        }
        if(output_enabled(OUTPUT_NORMALS_RGBA8)){
            outputs[params.output_offsets[OUTPUT_NORMALS_RGBA8] + u32(ctx.fb_offset)] = normal_to_rgba(normal);
        }
    }
    if((fc_output_mask & (15u << OUTPUT_FLOW_FLOAT32)) != 0u){
        write_flow(false, ctx, cam, direction, hit);
    }
}

@compute @workgroup_size(8, 4, 1)
fn main_auxiliary(@builtin(global_invocation_id) global_id: vec3<u32>){
    let ctx = pixel_launch_context(global_id.xy);
    if(!ctx.valid){
        return;
    }
    let cam = camera_close(ctx.cam_idx);
    let screen = (vec2<f32>(f32(ctx.local_x), f32(ctx.local_y)) + vec2<f32>(0.5)) / vec2<f32>(f32(params.cam_width), f32(params.cam_height));
    let direction = normalize(camera_dir_00(cam) + screen.x * camera_dir_du(cam) + screen.y * camera_dir_dv(cam));
    let hit = trace_closest_composed_region(0u, ctx.cam_idx, camera_pos(cam), direction, 0.0, 1e30);
    write_auxiliary(ctx, cam, direction, hit);
}

@compute @workgroup_size(8, 4, 1)
fn main_flow_forward(@builtin(global_invocation_id) global_id: vec3<u32>){
    let ctx = pixel_launch_context(global_id.xy);
    if(!ctx.valid){
        return;
    }
    let cam = camera_previous(ctx.cam_idx);
    let screen = (vec2<f32>(f32(ctx.local_x), f32(ctx.local_y)) + vec2<f32>(0.5)) / vec2<f32>(f32(params.cam_width), f32(params.cam_height));
    let direction = normalize(camera_dir_00(cam) + screen.x * camera_dir_du(cam) + screen.y * camera_dir_dv(cam));
    let previous_region = 1u + select(0u, u32(fc_motion_samples), fc_dynamic_motion_blur);
    let hit = trace_closest_composed_region(previous_region, ctx.cam_idx, camera_pos(cam), direction, 0.0, 1e30);
    write_flow(true, ctx, cam, direction, hit);
}

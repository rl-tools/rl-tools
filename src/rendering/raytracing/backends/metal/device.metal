// Metal device programs for the RLtools raytracing renderer.
// This file is a function-for-function mirror of backends/optix/device_impl.h — keep the shading
// logic in sync; the golden-image parity suite is the drift detector.
// Struct layouts (LaunchParams, MeshRecord) and buffer/function-constant indices must match
// include/rl_tools/rendering/raytracing/backends/metal/context.h.
#include <metal_stdlib>
#include <metal_raytracing>

using namespace metal;
using namespace metal::raytracing;

constant bool fc_srgb_output [[function_constant(0)]];
constant bool fc_motion_blur [[function_constant(1)]];
constant int fc_motion_samples [[function_constant(2)]];
constant int fc_aa_grid [[function_constant(3)]];
constant bool fc_checker_background [[function_constant(4)]];
constant bool fc_load_textures [[function_constant(5)]];
constant bool fc_normal_shading [[function_constant(6)]];
constant bool fc_metallic_reflections [[function_constant(7)]];
constant bool fc_pbr_shading [[function_constant(8)]];
constant bool fc_punctual_light_shadows [[function_constant(9)]];
constant int fc_overlay_count [[function_constant(10)]];
constant bool fc_semantic_segmentation [[function_constant(11)]];

struct LaunchParams{
    uint fb_width;
    uint fb_height;
    uint cam_width;
    uint cam_height;
    uint grid_cols;
    uint num_cameras;
    uint num_probes;
    uint num_scene_lights;
    float max_depth;
    float max_dist;
    packed_float3 ambient_color;
    packed_float3 miss_color_0;
    packed_float3 miss_color_1;
    float padding[3];
};
static_assert(sizeof(LaunchParams) == 88, "LaunchParams layout must match context.h");

struct MeshRecord{
    device const packed_int3* index;
    device const packed_float3* vertices;
    device const packed_float2* tex_coord;
    device const packed_float3* normal;
    texture2d<float> texture;
    texture2d<float> normal_map;
    texture2d<float> metallic_roughness_map;
    texture2d<float> emissive_map;
    texture2d<float> occlusion_map;
    packed_float3 color;
    float metallic;
    float roughness;
    float opacity;
    packed_float3 emissive;
    float alpha_cutoff;
    int alpha_mode;
    int has_texture;
    int has_normal_map;
    int has_metallic_roughness_map;
    int has_emissive_map;
    int has_occlusion_map;
    int padding[2];
};
static_assert(sizeof(MeshRecord) == 144, "MeshRecord layout must match context.h");

struct SceneLight{
    int type; // 0=directional, 1=point, 2=spot
    packed_float3 position;
    packed_float3 direction;
    packed_float3 color;
    float attenuation_constant;
    float attenuation_linear;
    float attenuation_quadratic;
    float cos_inner_cone;
    float cos_outer_cone;
};
static_assert(sizeof(SceneLight) == 60, "SceneLight layout must match rendering/raytracing/types.h");

struct Camera{
    packed_float3 pos;
    packed_float3 dir_00;
    packed_float3 dir_du;
    packed_float3 dir_dv;
};
static_assert(sizeof(Camera) == 48, "Camera layout must match rendering/raytracing/types.h");

struct InstanceData{
    float object_to_world[12]; // 3x4 row-major [R|t]
    float world_to_object[12];
    int identity;
    int padding[3];
};
static_assert(sizeof(InstanceData) == 112, "InstanceData layout must match context.h");

inline float3 transform_point(device const float* m, float3 p){
    return float3(m[0]*p.x + m[1]*p.y + m[2]*p.z + m[3],
                  m[4]*p.x + m[5]*p.y + m[6]*p.z + m[7],
                  m[8]*p.x + m[9]*p.y + m[10]*p.z + m[11]);
}

// normals transform with the inverse-transpose: multiply by the world_to_object columns
inline float3 transform_normal(device const float* m, float3 n){
    return float3(m[0]*n.x + m[4]*n.y + m[8]*n.z,
                  m[1]*n.x + m[5]*n.y + m[9]*n.z,
                  m[2]*n.x + m[6]*n.y + m[10]*n.z);
}

struct CollisionResult{
    float distance;
    int hit;
};
static_assert(sizeof(CollisionResult) == 8, "CollisionResult layout must match rendering/raytracing/types.h");

constexpr sampler tex_sampler(address::repeat, filter::linear, coord::normalized);

inline uint make_8bit(float f){
    return (uint)min(255, max(0, (int)(f * 256.0f)));
}

inline uint make_rgba(float3 color){
    return (make_8bit(color.x) << 0) | (make_8bit(color.y) << 8) | (make_8bit(color.z) << 16) | (0xffu << 24);
}

inline float linear_to_srgb(float x){
    if (x <= 0.0031308f) return 12.92f * x;
    return 1.055f * pow(x, 1.f / 2.4f) - 0.055f;
}

inline uint make_srgb_rgba_from_linear(float3 color){
    color.x = linear_to_srgb(fmin(fmax(color.x, 0.f), 1.f));
    color.y = linear_to_srgb(fmin(fmax(color.y, 0.f), 1.f));
    color.z = linear_to_srgb(fmin(fmax(color.z, 0.f), 1.f));
    return make_rgba(color);
}

inline uint make_linear_rgba_from_linear(float3 color){
    color.x = fmin(fmax(color.x, 0.f), 1.f);
    color.y = fmin(fmax(color.y, 0.f), 1.f);
    color.z = fmin(fmax(color.z, 0.f), 1.f);
    return make_rgba(color);
}

inline float3 lerp_camera_vec(float3 a, float3 b, float t){
    return (1.f - t) * a + t * b;
}

struct PixelLaunchContext{
    int cam_idx;
    int local_x;
    int local_y;
    int fb_offset;
    bool valid;
};

inline PixelLaunchContext pixel_launch_context(constant LaunchParams& params, uint2 pixel_id){
    const int tile_col = (int)(pixel_id.x / params.cam_width);
    const int tile_row = (int)(pixel_id.y / params.cam_height);
    const int cam_idx  = tile_row * (int)params.grid_cols + tile_col;
    const int local_x = (int)pixel_id.x - tile_col * (int)params.cam_width;
    const int local_y = (int)pixel_id.y - tile_row * (int)params.cam_height;
    const int fb_offset = cam_idx * (int)params.cam_width * (int)params.cam_height + local_y * (int)params.cam_width + local_x;
    return {cam_idx, local_x, local_y, fb_offset, cam_idx < (int)params.num_cameras};
}

struct OverlayStructure{
    instance_acceleration_structure structure;
    uint num_active;
    uint padding;
};

struct TraceContext{
    instance_acceleration_structure accel;
    device const MeshRecord* meshes;
    device const SceneLight* scene_lights;
    device const uint* instance_record_base;
    device const InstanceData* instance_data;
    device const OverlayStructure* overlays;
    device const uint* attachments;
    constant LaunchParams* params;
    uint2 pixel_id;
    int camera;
};

// min-t composition of the shared world and the overlays attached to this camera; at
// fc_overlay_count == 0 the loop specializes away and this is a plain intersect
inline intersection_result<triangle_data, instancing> intersect_composed(
    instance_acceleration_structure accel,
    device const OverlayStructure* overlays,
    device const uint* attachments,
    int camera,
    ray r,
    bool accept_any)
{
    intersector<triangle_data, instancing> i;
    i.assume_geometry_type(geometry_type::triangle);
    i.force_opacity(forced_opacity::opaque);
    i.accept_any_intersection(accept_any);
    intersection_result<triangle_data, instancing> best = i.intersect(r, accel);
    for (int k = 0; k < fc_overlay_count; k++) {
        if (accept_any && best.type != intersection_type::none) {
            return best;
        }
        const uint overlay = attachments[camera * fc_overlay_count + k];
        if (overlay == 0xFFFFFFFFu) continue;
        device const OverlayStructure& entry = overlays[overlay];
        if (entry.num_active == 0) continue;
        ray clamped = r;
        if (best.type != intersection_type::none) {
            clamped.max_distance = best.distance;
        }
        intersection_result<triangle_data, instancing> hit = i.intersect(clamped, entry.structure);
        if (hit.type != intersection_type::none) {
            best = hit;
        }
    }
    return best;
}

inline float3 miss_color(thread const TraceContext& ctx){
    if (fc_checker_background) {
        int checker_pattern = ((int)ctx.pixel_id.x / 8) ^ ((int)ctx.pixel_id.y / 8);
        return (checker_pattern & 1) ? float3(ctx.params->miss_color_1) : float3(ctx.params->miss_color_0);
    }
    return float3(ctx.params->miss_color_0);
}

inline bool trace_shadow_occluded(thread const TraceContext& ctx, float3 pos, float3 direction, float max_dist){
    ray r(pos, direction, 1e-3f, max_dist);
    intersection_result<triangle_data, instancing> hit = intersect_composed(ctx.accel, ctx.overlays, ctx.attachments, ctx.camera, r, true);
    return hit.type != intersection_type::none;
}

inline float trace_depth_distance(instance_acceleration_structure accel, device const OverlayStructure* overlays, device const uint* attachments, int camera, float3 pos, float3 direction, float max_depth){
    ray r(pos, direction, 0.f, max_depth);
    intersection_result<triangle_data, instancing> hit = intersect_composed(accel, overlays, attachments, camera, r, false);
    return hit.type == intersection_type::none ? max_depth : hit.distance;
}

template <int DEPTH>
struct RGBTracer;

// Bounded depth-1 "recursion": the ENABLED=false specialization does not reference RGBTracer, so
// template instantiation terminates (mirrors the optixGetPayload_2()-based depth in device_impl.h).
template <int DEPTH, bool ENABLED>
struct SecondaryTrace{
    static float3 trace(thread const TraceContext& ctx, float3 origin, float3 direction, float tmin){
        return RGBTracer<DEPTH>::trace(ctx, origin, direction, tmin, 1e20f);
    }
};

template <int DEPTH>
struct SecondaryTrace<DEPTH, false>{
    static float3 trace(thread const TraceContext&, float3, float3, float){
        return float3(0.f);
    }
};

template <int DEPTH>
inline float3 shade_basic(thread const TraceContext& ctx, thread const ray& r, thread const intersection_result<triangle_data, instancing>& hit){
    constexpr bool SECONDARY = DEPTH < 1;
    device const MeshRecord& self = ctx.meshes[ctx.instance_record_base[hit.user_instance_id] + hit.geometry_id];
    device const InstanceData& instance = ctx.instance_data[hit.user_instance_id];

    float3 base_color = float3(self.color);
    float3 normal_geometric = float3(0.f, 0.f, 1.f);
    float3 ray_dir = float3(0.f);

    if (fc_load_textures || fc_normal_shading || fc_metallic_reflections) {
        const int prim_id = hit.primitive_id;
        const int3 index = int3(self.index[prim_id]);

        if (fc_normal_shading || fc_metallic_reflections) {
            float3 vertex_a = float3(self.vertices[index.x]);
            float3 vertex_b = float3(self.vertices[index.y]);
            float3 vertex_c = float3(self.vertices[index.z]);
            if (!instance.identity) {
                vertex_a = transform_point(instance.object_to_world, vertex_a);
                vertex_b = transform_point(instance.object_to_world, vertex_b);
                vertex_c = transform_point(instance.object_to_world, vertex_c);
            }
            normal_geometric = normalize(cross(vertex_b - vertex_a, vertex_c - vertex_a));
            ray_dir = r.direction;
        }

        if (fc_load_textures) {
            if (self.has_texture && self.tex_coord != nullptr) {
                const float2 bary = hit.triangle_barycentric_coord;
                const float2 tc
                    = (1.f - bary.x - bary.y) * float2(self.tex_coord[index.x])
                    +        bary.x           * float2(self.tex_coord[index.y])
                    +               bary.y    * float2(self.tex_coord[index.z]);
                float4 tex_color = self.texture.sample(tex_sampler, tc);
                base_color = tex_color.xyz * float3(self.color);
            }
        }
    }

    float3 direct = base_color;
    if (fc_normal_shading) {
        direct = (.2f + .8f * fabs(dot(ray_dir, normal_geometric))) * base_color;
    }

    if (fc_metallic_reflections) {
        if (SECONDARY && self.metallic > 0.f) {
            float3 hit_point = r.origin + ray_dir * hit.distance;
            float3 n = dot(ray_dir, normal_geometric) > 0.f ? -normal_geometric : normal_geometric;
            float3 reflect_dir = ray_dir - 2.f * dot(ray_dir, n) * n;

            float3 reflected_color = SecondaryTrace<DEPTH + 1, SECONDARY>::trace(ctx, hit_point, reflect_dir, 1e-3f);

            float cos_theta = fabs(dot(ray_dir, n));
            float fresnel = self.metallic * (0.04f + 0.96f * pow(1.f - cos_theta, 5.f));
            return direct * ((1.f - fresnel) + fresnel * reflected_color);
        }
        return direct;
    }
    return direct;
}

template <int DEPTH>
inline float3 shade_pbr(thread const TraceContext& ctx, thread const ray& r, thread const intersection_result<triangle_data, instancing>& hit){
    constexpr bool SECONDARY = DEPTH < 1;
    device const MeshRecord& self = ctx.meshes[ctx.instance_record_base[hit.user_instance_id] + hit.geometry_id];
    device const InstanceData& instance = ctx.instance_data[hit.user_instance_id];

    const int prim_id = hit.primitive_id;
    const int3 index = int3(self.index[prim_id]);
    float3 vertex_a = float3(self.vertices[index.x]);
    float3 vertex_b = float3(self.vertices[index.y]);
    float3 vertex_c = float3(self.vertices[index.z]);
    if (!instance.identity) {
        vertex_a = transform_point(instance.object_to_world, vertex_a);
        vertex_b = transform_point(instance.object_to_world, vertex_b);
        vertex_c = transform_point(instance.object_to_world, vertex_c);
    }
    const float2 bary = hit.triangle_barycentric_coord;
    const float w0 = 1.f - bary.x - bary.y;

    const float3 edge1 = vertex_b - vertex_a;
    const float3 edge2 = vertex_c - vertex_a;
    const float3 normal_geometric = normalize(cross(edge1, edge2));

    float3 N;
    if (self.normal != nullptr) {
        float3 normal_interpolated = w0 * float3(self.normal[index.x]) + bary.x * float3(self.normal[index.y]) + bary.y * float3(self.normal[index.z]);
        if (!instance.identity) {
            normal_interpolated = transform_normal(instance.world_to_object, normal_interpolated);
        }
        N = normalize(normal_interpolated);
    } else {
        N = normal_geometric;
    }

    const float3 ray_dir = r.direction;
    if (dot(ray_dir, N) > 0.f) N = -N;

    const float2 tc = (self.tex_coord != nullptr)
        ? w0 * float2(self.tex_coord[index.x]) + bary.x * float2(self.tex_coord[index.y]) + bary.y * float2(self.tex_coord[index.z])
        : float2(0.f);

    float3 base_color = float3(self.color);
    float alpha = self.opacity;
    if (self.has_texture && self.tex_coord != nullptr) {
        float4 tex_color = self.texture.sample(tex_sampler, tc);
        base_color = tex_color.xyz * float3(self.color);
        alpha *= tex_color.w;
    }

    float metallic = self.metallic;
    float roughness = self.roughness;
    if (self.has_metallic_roughness_map && self.tex_coord != nullptr) {
        float4 mr_sample = self.metallic_roughness_map.sample(tex_sampler, tc);
        roughness = mr_sample.y * self.roughness;
        metallic = mr_sample.z * self.metallic;
    }

    if (self.has_normal_map && self.tex_coord != nullptr) {
        const float2 tc0 = float2(self.tex_coord[index.x]);
        const float2 tc1 = float2(self.tex_coord[index.y]);
        const float2 tc2 = float2(self.tex_coord[index.z]);
        const float2 duv1 = tc1 - tc0;
        const float2 duv2 = tc2 - tc0;
        float det = duv1.x * duv2.y - duv2.x * duv1.y;
        if (fabs(det) > 1e-8f) {
            float inv_det = 1.f / det;
            float3 T = normalize(inv_det * (duv2.y * edge1 - duv1.y * edge2));
            T = normalize(T - dot(T, N) * N);
            float3 B = cross(N, T);
            float4 nm_sample = self.normal_map.sample(tex_sampler, float2(w0 * tc0.x + bary.x * tc1.x + bary.y * tc2.x, w0 * tc0.y + bary.x * tc1.y + bary.y * tc2.y));
            float3 n_tangent = float3(nm_sample.x * 2.f - 1.f, -(nm_sample.y * 2.f - 1.f), nm_sample.z * 2.f - 1.f);
            N = normalize(T * n_tangent.x + B * n_tangent.y + N * n_tangent.z);
        }
    }

    roughness = fmax(roughness, 0.04f);
    float roughness_alpha = roughness * roughness;
    float alpha2 = roughness_alpha * roughness_alpha;
    float k = (roughness + 1.f) * (roughness + 1.f) / 8.f;

    float3 V = -ray_dir;
    float NdotV = fmax(dot(N, V), 1e-4f);
    float3 F0 = float3(0.04f) * (1.f - metallic) + base_color * metallic;

    float3 hit_point = r.origin + ray_dir * hit.distance;

    float3 Lo = float3(0.f);
    for (int li = 0; li < (int)ctx.params->num_scene_lights; li++) {
        device const SceneLight& light = ctx.scene_lights[li];
        float3 Lc = float3(light.color);
        float3 L;
        float attenuation = 1.f;
        float light_distance = 1e20f;

        if (light.type == 0) {
            L = float3(light.direction);
        } else {
            float3 to_light = float3(light.position) - hit_point;
            float dist = length(to_light);
            light_distance = fmax(dist - 1e-3f, 0.f);
            L = to_light * (1.f / fmax(dist, 1e-6f));
            attenuation = 1.f / (light.attenuation_constant + light.attenuation_linear * dist + light.attenuation_quadratic * dist * dist);
            if (light.type == 2) {
                float3 spot_dir = float3(light.direction);
                float cos_angle = dot(-L, spot_dir);
                float denom = light.cos_inner_cone - light.cos_outer_cone;
                float spot_t = (cos_angle - light.cos_outer_cone) / (fabs(denom) > 1e-6f ? denom : 1e-6f);
                attenuation *= fmax(fmin(spot_t, 1.f), 0.f);
            }
        }

        float NdotL = fmax(dot(N, L), 0.f);
        if (NdotL <= 0.f) continue;
        if (fc_punctual_light_shadows) {
            if (light.type != 0 && trace_shadow_occluded(ctx, hit_point + N * 2e-3f, L, light_distance)) continue;
        }

        float3 H = normalize(V + L);
        float NdotH = fmax(dot(N, H), 0.f);
        float VdotH = fmax(dot(V, H), 0.f);

        float denom_D = NdotH * NdotH * (alpha2 - 1.f) + 1.f;
        float D = alpha2 / (3.14159265f * denom_D * denom_D);

        float G1_V = NdotV / (NdotV * (1.f - k) + k);
        float G1_L = NdotL / (NdotL * (1.f - k) + k);
        float G = G1_V * G1_L;

        float pow5 = pow(1.f - VdotH, 5.f);
        float3 F = F0 + (float3(1.f) - F0) * pow5;

        float3 specular = D * G * F * (1.f / (4.f * NdotV * NdotL + 1e-4f));
        float3 kd = (float3(1.f) - F) * (1.f - metallic);
        float3 diffuse = kd * base_color * (1.f / 3.14159265f);

        Lo = Lo + (diffuse + specular) * Lc * (attenuation * NdotL);
    }

    float occlusion = 1.f;
    if (self.has_occlusion_map && self.tex_coord != nullptr) {
        float4 ao_sample = self.occlusion_map.sample(tex_sampler, tc);
        occlusion = ao_sample.x;
    }

    float3 emissive_color = float3(0.f);
    if (self.has_emissive_map && self.tex_coord != nullptr) {
        float4 em_sample = self.emissive_map.sample(tex_sampler, tc);
        emissive_color = em_sample.xyz * float3(self.emissive);
    } else {
        emissive_color = float3(self.emissive);
    }

    float3 ambient = float3(ctx.params->ambient_color) * base_color * ((1.f - metallic) * occlusion);
    float3 color = ambient + Lo + emissive_color;

    if (SECONDARY && metallic > 0.1f) {
        float3 reflect_dir = ray_dir - 2.f * dot(ray_dir, N) * N;

        float3 reflected_color = SecondaryTrace<DEPTH + 1, SECONDARY>::trace(ctx, hit_point, reflect_dir, 1e-3f);

        float fresnel_refl = F0.x + (1.f - F0.x) * pow(1.f - fmax(dot(V, N), 0.f), 5.f);
        float reflection_weight = fresnel_refl * (1.f - roughness);
        color = color * (1.f - reflection_weight) + reflected_color * reflection_weight;
    }

    bool transparent = (self.alpha_mode == 2 && alpha < 0.99f) || (self.alpha_mode == 1 && alpha < self.alpha_cutoff);
    if (SECONDARY && transparent) {
        float3 behind_color = SecondaryTrace<DEPTH + 1, SECONDARY>::trace(ctx, hit_point, ray_dir, 1e-5f);

        if (self.alpha_mode == 1) {
            color = behind_color;
        } else {
            color = color * alpha + behind_color * (1.f - alpha);
        }
    }

    return color;
}

template <int DEPTH>
struct RGBTracer{
    static float3 trace(thread const TraceContext& ctx, float3 origin, float3 direction, float tmin, float tmax){
        ray r(origin, direction, tmin, tmax);
        intersection_result<triangle_data, instancing> hit = intersect_composed(ctx.accel, ctx.overlays, ctx.attachments, ctx.camera, r, false);
        if (hit.type == intersection_type::none) {
            return miss_color(ctx);
        }
        if (fc_pbr_shading) {
            return shade_pbr<DEPTH>(ctx, r, hit);
        }
        return shade_basic<DEPTH>(ctx, r, hit);
    }
};

kernel void render_rgb(
    constant LaunchParams& params [[buffer(0)]],
    device const Camera* cameras_close [[buffer(1)]],
    device const Camera* cameras_open [[buffer(2)]],
    device uint* fb [[buffer(3)]],
    device const MeshRecord* meshes [[buffer(4)]],
    device const SceneLight* scene_lights [[buffer(5)]],
    instance_acceleration_structure accel [[buffer(8)]],
    device const uint* instance_record_base [[buffer(9)]],
    device const InstanceData* instance_data [[buffer(10)]],
    device const OverlayStructure* overlays [[buffer(11)]],
    device const uint* overlay_attachments [[buffer(12)]],
    uint2 pixel_id [[thread_position_in_grid]])
{
    const PixelLaunchContext ctx = pixel_launch_context(params, pixel_id);
    if (!ctx.valid)
        return;

    TraceContext trace_ctx{accel, meshes, scene_lights, instance_record_base, instance_data, overlays, overlay_attachments, &params, pixel_id, ctx.cam_idx};

    float3 accumulated = float3(0.f);
    const float inv_aa_grid = 1.f / float(fc_aa_grid);
    for (int motion_i = 0; motion_i < fc_motion_samples; motion_i++) {
        float3 pos;
        float3 dir_00;
        float3 dir_du;
        float3 dir_dv;
        if (fc_motion_blur) {
            device const Camera& cam_open = cameras_open[ctx.cam_idx];
            device const Camera& cam_close = cameras_close[ctx.cam_idx];
            const float shutter_t = (float(motion_i) + .5f) * (1.f / float(fc_motion_samples));
            pos = lerp_camera_vec(float3(cam_open.pos), float3(cam_close.pos), shutter_t);
            dir_00 = lerp_camera_vec(float3(cam_open.dir_00), float3(cam_close.dir_00), shutter_t);
            dir_du = lerp_camera_vec(float3(cam_open.dir_du), float3(cam_close.dir_du), shutter_t);
            dir_dv = lerp_camera_vec(float3(cam_open.dir_dv), float3(cam_close.dir_dv), shutter_t);
        }
        else {
            device const Camera& cam = cameras_close[ctx.cam_idx];
            pos = float3(cam.pos);
            dir_00 = float3(cam.dir_00);
            dir_du = float3(cam.dir_du);
            dir_dv = float3(cam.dir_dv);
        }
        for (int aa_y = 0; aa_y < fc_aa_grid; aa_y++) {
            for (int aa_x = 0; aa_x < fc_aa_grid; aa_x++) {
                const float2 screen = (float2(ctx.local_x, ctx.local_y) + float2((float(aa_x) + .5f) * inv_aa_grid, (float(aa_y) + .5f) * inv_aa_grid)) / float2(params.cam_width, params.cam_height);
                const float3 direction = normalize(dir_00 + screen.x * dir_du + screen.y * dir_dv);
                accumulated += RGBTracer<0>::trace(trace_ctx, pos, direction, 0.f, 1e30f);
            }
        }
    }
    const int samples = fc_motion_samples * fc_aa_grid * fc_aa_grid;
    const float3 color = accumulated * (1.f / float(samples));
    if (fc_srgb_output) {
        fb[ctx.fb_offset] = make_srgb_rgba_from_linear(color);
    }
    else {
        fb[ctx.fb_offset] = make_linear_rgba_from_linear(color);
    }
}

kernel void render_depth(
    constant LaunchParams& params [[buffer(0)]],
    device const Camera* cameras_close [[buffer(1)]],
    device const Camera* cameras_open [[buffer(2)]],
    device float* depth_out [[buffer(3)]],
    instance_acceleration_structure accel [[buffer(8)]],
    device const OverlayStructure* overlays [[buffer(11)]],
    device const uint* overlay_attachments [[buffer(12)]],
    uint2 pixel_id [[thread_position_in_grid]])
{
    const PixelLaunchContext ctx = pixel_launch_context(params, pixel_id);
    if (!ctx.valid)
        return;

    float accumulated = 0.f;
    const float inv_aa_grid = 1.f / float(fc_aa_grid);
    for (int motion_i = 0; motion_i < fc_motion_samples; motion_i++) {
        float3 pos;
        float3 dir_00;
        float3 dir_du;
        float3 dir_dv;
        if (fc_motion_blur) {
            device const Camera& cam_open = cameras_open[ctx.cam_idx];
            device const Camera& cam_close = cameras_close[ctx.cam_idx];
            const float shutter_t = (float(motion_i) + .5f) * (1.f / float(fc_motion_samples));
            pos = lerp_camera_vec(float3(cam_open.pos), float3(cam_close.pos), shutter_t);
            dir_00 = lerp_camera_vec(float3(cam_open.dir_00), float3(cam_close.dir_00), shutter_t);
            dir_du = lerp_camera_vec(float3(cam_open.dir_du), float3(cam_close.dir_du), shutter_t);
            dir_dv = lerp_camera_vec(float3(cam_open.dir_dv), float3(cam_close.dir_dv), shutter_t);
        }
        else {
            device const Camera& cam = cameras_close[ctx.cam_idx];
            pos = float3(cam.pos);
            dir_00 = float3(cam.dir_00);
            dir_du = float3(cam.dir_du);
            dir_dv = float3(cam.dir_dv);
        }
        for (int aa_y = 0; aa_y < fc_aa_grid; aa_y++) {
            for (int aa_x = 0; aa_x < fc_aa_grid; aa_x++) {
                const float2 screen = (float2(ctx.local_x, ctx.local_y) + float2((float(aa_x) + .5f) * inv_aa_grid, (float(aa_y) + .5f) * inv_aa_grid)) / float2(params.cam_width, params.cam_height);
                const float3 direction = normalize(dir_00 + screen.x * dir_du + screen.y * dir_dv);
                accumulated += trace_depth_distance(accel, overlays, overlay_attachments, ctx.cam_idx, pos, direction, params.max_depth);
            }
        }
    }
    const int samples = fc_motion_samples * fc_aa_grid * fc_aa_grid;
    depth_out[ctx.fb_offset] = accumulated * (1.f / float(samples));
}

// single-sample by design: instance labels cannot be averaged, so anti-aliasing and motion blur
// do not apply (shutter-close camera, pixel-center ray)
kernel void render_segmentation(
    constant LaunchParams& params [[buffer(0)]],
    device const Camera* cameras_close [[buffer(1)]],
    device uint* segmentation_out [[buffer(3)]],
    instance_acceleration_structure accel [[buffer(8)]],
    device const OverlayStructure* overlays [[buffer(11)]],
    device const uint* overlay_attachments [[buffer(12)]],
    device const uint* instance_classes [[buffer(13)]],
    uint2 pixel_id [[thread_position_in_grid]])
{
    const PixelLaunchContext ctx = pixel_launch_context(params, pixel_id);
    if (!ctx.valid)
        return;

    device const Camera& cam = cameras_close[ctx.cam_idx];
    const float2 screen = (float2(ctx.local_x, ctx.local_y) + float2(0.5f, 0.5f)) / float2(params.cam_width, params.cam_height);
    const float3 direction = normalize(float3(cam.dir_00) + screen.x * float3(cam.dir_du) + screen.y * float3(cam.dir_dv));

    ray r(float3(cam.pos), direction, 0.f, 1e30f);
    intersection_result<triangle_data, instancing> hit = intersect_composed(accel, overlays, overlay_attachments, ctx.cam_idx, r, false);
    segmentation_out[ctx.fb_offset] = hit.type == intersection_type::none ? 0xFFFFFFFFu : (fc_semantic_segmentation ? instance_classes[hit.user_instance_id] : (uint)hit.user_instance_id);
}

kernel void render_collision(
    constant LaunchParams& params [[buffer(0)]],
    device const Camera* cameras [[buffer(1)]],
    device const packed_float3* probe_directions [[buffer(6)]],
    device CollisionResult* results [[buffer(7)]],
    instance_acceleration_structure accel [[buffer(8)]],
    device const OverlayStructure* overlays [[buffer(11)]],
    device const uint* overlay_attachments [[buffer(12)]],
    uint2 idx [[thread_position_in_grid]])
{
    const int cam_idx   = (int)idx.x;
    const int probe_idx = (int)idx.y;

    if (cam_idx >= (int)params.num_cameras || probe_idx >= (int)params.num_probes)
        return;

    device const Camera& cam = cameras[cam_idx];

    float3 dir;
    if (probe_idx == 0) {
        dir = normalize(float3(cam.dir_00) + 0.5f * float3(cam.dir_du) + 0.5f * float3(cam.dir_dv));
    } else {
        dir = float3(probe_directions[probe_idx]);
    }

    ray r(float3(cam.pos), dir, 1e-3f, params.max_dist);
    intersection_result<triangle_data, instancing> hit = intersect_composed(accel, overlays, overlay_attachments, cam_idx, r, false);

    CollisionResult result;
    if (hit.type == intersection_type::none) {
        result.distance = params.max_dist;
        result.hit = 0;
    } else {
        result.distance = hit.distance;
        result.hit = 1;
    }
    results[cam_idx * (int)params.num_probes + probe_idx] = result;
}

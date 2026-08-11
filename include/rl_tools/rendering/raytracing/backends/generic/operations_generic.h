#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_GENERIC_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_GENERIC_OPERATIONS_GENERIC_H

// Freestanding software raytracing kernels. This file is a mirror of the device programs in
// backends/optix/device_impl.h and backends/metal/device.metal — keep the shading logic in sync;
// the cross-backend parity comparison is the drift detector. SPEC is duck-typed (this header must
// not include renderer.h, which is not freestanding); the host adapter in operations_cpu.h passes
// rendering::raytracing::Specification.
#include "../../types.h"
#include "../../../../utils/generic/typing.h"

#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering::raytracing::backends::generic{
        template <typename T>
        struct Vec3{
            T x, y, z;
        };
        template <typename T> RL_TOOLS_FUNCTION_PLACEMENT Vec3<T> operator+(Vec3<T> a, Vec3<T> b){ return {a.x + b.x, a.y + b.y, a.z + b.z}; }
        template <typename T> RL_TOOLS_FUNCTION_PLACEMENT Vec3<T> operator-(Vec3<T> a, Vec3<T> b){ return {a.x - b.x, a.y - b.y, a.z - b.z}; }
        template <typename T> RL_TOOLS_FUNCTION_PLACEMENT Vec3<T> operator*(Vec3<T> a, Vec3<T> b){ return {a.x * b.x, a.y * b.y, a.z * b.z}; }
        template <typename T> RL_TOOLS_FUNCTION_PLACEMENT Vec3<T> operator*(T s, Vec3<T> v){ return {s * v.x, s * v.y, s * v.z}; }
        template <typename T> RL_TOOLS_FUNCTION_PLACEMENT Vec3<T> operator*(Vec3<T> v, T s){ return {v.x * s, v.y * s, v.z * s}; }
        template <typename T> RL_TOOLS_FUNCTION_PLACEMENT Vec3<T> operator-(Vec3<T> v){ return {-v.x, -v.y, -v.z}; }
        template <typename T> RL_TOOLS_FUNCTION_PLACEMENT T dot(Vec3<T> a, Vec3<T> b){ return a.x * b.x + a.y * b.y + a.z * b.z; }
        template <typename T> RL_TOOLS_FUNCTION_PLACEMENT Vec3<T> cross(Vec3<T> a, Vec3<T> b){
            return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
        }
        template <typename T> RL_TOOLS_FUNCTION_PLACEMENT Vec3<T> to_vec3(const T v[3]){ return {v[0], v[1], v[2]}; }
        template <typename MATH, typename T> RL_TOOLS_FUNCTION_PLACEMENT T length(const MATH& math_device, Vec3<T> v){
            return math::sqrt(math_device, dot(v, v));
        }
        template <typename MATH, typename T> RL_TOOLS_FUNCTION_PLACEMENT Vec3<T> normalize(const MATH& math_device, Vec3<T> v){
            const T inv_length = (T)1 / math::sqrt(math_device, dot(v, v));
            return v * inv_length;
        }
        template <typename T> RL_TOOLS_FUNCTION_PLACEMENT T clamp01(T x){ return x < (T)0 ? (T)0 : (x > (T)1 ? (T)1 : x); }
        template <typename T> RL_TOOLS_FUNCTION_PLACEMENT T minimum(T a, T b){ return a < b ? a : b; }
        template <typename T> RL_TOOLS_FUNCTION_PLACEMENT T maximum(T a, T b){ return a > b ? a : b; }

        template <typename TI>
        struct TextureView{
            const unsigned char* pixels = nullptr; // RGBA8; nullptr => absent
            TI width = 0;
            TI height = 0;
        };

        template <typename T, typename TI>
        struct MeshView{
            const int* indices = nullptr;      // 3 per triangle
            const T* vertices = nullptr;       // 3 per vertex
            const T* tex_coords = nullptr;     // 2 per vertex (nullptr if absent)
            const T* normals = nullptr;        // 3 per vertex (nullptr if absent)
            TextureView<TI> texture;
            TextureView<TI> normal_map;
            TextureView<TI> metallic_roughness_map;
            TextureView<TI> emissive_map;
            TextureView<TI> occlusion_map;
            T color[3] = {0, 0, 0};
            T emissive[3] = {0, 0, 0};
            T metallic = 0;
            T roughness = 1;
            T opacity = 1;
            T alpha_cutoff = (T)0.5;
            int alpha_mode = 0;
        };

        template <typename T, typename TI>
        struct BVHNode{
            T bounds_min[3];
            T bounds_max[3];
            TI left_or_first; // count == 0: index of left child (right = left + 1); count > 0: first index into primitives
            TI count;
        };

        // BLAS of one object: a BVH over a contiguous range of scene-global triangle ids
        template <typename T, typename TI>
        struct ObjectView{
            const BVHNode<T, TI>* nodes = nullptr;
            const TI* primitives = nullptr; // leaf permutation of scene-global triangle indices
            TI num_nodes = 0;
        };

        template <typename T, typename TI>
        struct InstanceView{
            TI object = 0;
            T object_to_world[12]; // 3x4 row-major [R|t]
            T world_to_object[12];
            bool identity = true;
        };

        // dynamic overlay: a tiny TLAS whose leaf primitives are GLOBAL instance indices into
        // SceneView::instances, so BLAS traversal and shading work on overlays verbatim
        template <typename T, typename TI>
        struct OverlayView{
            const BVHNode<T, TI>* tlas_nodes = nullptr;
            const TI* tlas_primitives = nullptr;
            TI num_tlas_nodes = 0;
        };

        template <typename T, typename TI>
        struct SceneView{
            const MeshView<T, TI>* meshes = nullptr;
            TI num_meshes = 0;
            const TI* triangle_mesh = nullptr;  // global triangle -> mesh index
            const TI* triangle_local = nullptr; // global triangle -> local primitive index within the mesh
            TI num_triangles = 0;
            const ObjectView<T, TI>* objects = nullptr;
            TI num_objects = 0;
            const InstanceView<T, TI>* instances = nullptr;
            TI num_instances = 0;
            const BVHNode<T, TI>* tlas_nodes = nullptr;
            const TI* tlas_primitives = nullptr; // TLAS leaf permutation of instance indices
            TI num_tlas_nodes = 0;
            const OverlayView<T, TI>* overlays = nullptr;
            TI num_overlays = 0;
            const TI* attachments = nullptr; // NUM_CAMERAS * MAX_OVERLAYS_PER_CAMERA, ~0 = empty
            const SceneLight* lights = nullptr;
            TI num_lights = 0;
            const T* probe_directions = nullptr; // 3 per probe
            T ambient_color[3] = {0, 0, 0};
            T miss_color_0[3] = {0, 0, 0};
            T miss_color_1[3] = {0, 0, 0};
            T max_depth = 0;
            T max_dist = 0;
            const Camera<T>* cameras_close = nullptr;
            const Camera<T>* cameras_open = nullptr;
            unsigned int* frame_buffer = nullptr;
            float* depth_buffer = nullptr;
            unsigned int* segmentation_buffer = nullptr;
            float* normals_buffer = nullptr; // 3 per pixel, world-frame unit normal or zero on miss
            float* flow_buffer = nullptr;    // 2 per pixel, backward flow of the shutter-close frame in pixels
            const float* flow_deltas = nullptr; // 12 per overlay slot: world_open ∘ world_close⁻¹
            TI first_overlay_instance = 0;   // global instance ids >= this index the flow-delta table
            const unsigned int* instance_classes = nullptr; // indexed by global instance id
            CollisionResult* collision_results = nullptr;
            float* observation = nullptr; // 3 per pixel, written pre-quantization when set
            float* rgb_accumulation = nullptr;   // 3 per pixel, linear radiance summed across dynamic-motion-blur passes
            float* depth_accumulation = nullptr; // 1 per pixel
        };

        namespace constants{
            template <typename TI> constexpr TI LEAF_SIZE = 4;
            template <typename TI> constexpr TI BUILD_STACK_SIZE = 256;
            template <typename TI> constexpr TI TRAVERSAL_STACK_SIZE = 96;
        }

        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT unsigned int make_8bit(T f){
            const int value = (int)(f * (T)256);
            return (unsigned int)(value < 0 ? 0 : (value > 255 ? 255 : value));
        }

        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT unsigned int make_rgba(Vec3<T> color){
            return (make_8bit(color.x) << 0) | (make_8bit(color.y) << 8) | (make_8bit(color.z) << 16) | (0xffu << 24);
        }

        template <typename MATH, typename T>
        RL_TOOLS_FUNCTION_PLACEMENT T linear_to_srgb(const MATH& math_device, T x){
            if (x <= (T)0.0031308) return (T)12.92 * x;
            return (T)1.055 * math::pow(math_device, x, (T)1 / (T)2.4) - (T)0.055;
        }

        template <typename MATH, typename T>
        RL_TOOLS_FUNCTION_PLACEMENT unsigned int make_srgb_rgba_from_linear(const MATH& math_device, Vec3<T> color){
            color.x = linear_to_srgb(math_device, clamp01(color.x));
            color.y = linear_to_srgb(math_device, clamp01(color.y));
            color.z = linear_to_srgb(math_device, clamp01(color.z));
            return make_rgba(color);
        }

        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT unsigned int make_linear_rgba_from_linear(Vec3<T> color){
            color.x = clamp01(color.x);
            color.y = clamp01(color.y);
            color.z = clamp01(color.z);
            return make_rgba(color);
        }

        template <typename MATH, typename T>
        RL_TOOLS_FUNCTION_PLACEMENT T srgb_texel_to_linear(const MATH& math_device, unsigned char value){
            const T x = (T)value / (T)255;
            return x <= (T)0.04045 ? x / (T)12.92 : math::pow(math_device, (x + (T)0.055) / (T)1.055, (T)2.4);
        }

        // GPU-style normalized-coordinate bilinear sampling with repeat wrap; sRGB decode happens
        // per texel before filtering (matching the texture units of the other backends).
        template <typename MATH, typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void sample_texture(const MATH& math_device, const TextureView<TI>& texture, T u, T v, bool srgb, T out[4]){
            const int width = (int)texture.width;
            const int height = (int)texture.height;
            const T x = u * (T)width - (T)0.5;
            const T y = v * (T)height - (T)0.5;
            int x0 = (int)x; if((T)x0 > x) x0--;
            int y0 = (int)y; if((T)y0 > y) y0--;
            const T fx = x - (T)x0;
            const T fy = y - (T)y0;
            const int x0_wrapped = ((x0 % width) + width) % width;
            const int x1_wrapped = (((x0 + 1) % width) + width) % width;
            const int y0_wrapped = ((y0 % height) + height) % height;
            const int y1_wrapped = (((y0 + 1) % height) + height) % height;
            const unsigned char* texel_00 = &texture.pixels[((TI)y0_wrapped * (TI)width + (TI)x0_wrapped) * 4];
            const unsigned char* texel_10 = &texture.pixels[((TI)y0_wrapped * (TI)width + (TI)x1_wrapped) * 4];
            const unsigned char* texel_01 = &texture.pixels[((TI)y1_wrapped * (TI)width + (TI)x0_wrapped) * 4];
            const unsigned char* texel_11 = &texture.pixels[((TI)y1_wrapped * (TI)width + (TI)x1_wrapped) * 4];
            for(int channel = 0; channel < 4; channel++){
                const bool decode = srgb && channel < 3;
                const T value_00 = decode ? srgb_texel_to_linear<MATH, T>(math_device, texel_00[channel]) : (T)texel_00[channel] / (T)255;
                const T value_10 = decode ? srgb_texel_to_linear<MATH, T>(math_device, texel_10[channel]) : (T)texel_10[channel] / (T)255;
                const T value_01 = decode ? srgb_texel_to_linear<MATH, T>(math_device, texel_01[channel]) : (T)texel_01[channel] / (T)255;
                const T value_11 = decode ? srgb_texel_to_linear<MATH, T>(math_device, texel_11[channel]) : (T)texel_11[channel] / (T)255;
                const T top = value_00 + fx * (value_10 - value_00);
                const T bottom = value_01 + fx * (value_11 - value_01);
                out[channel] = top + fy * (bottom - top);
            }
        }

        template <typename T, typename TI>
        struct Hit{
            T t;
            T u;
            T v;
            TI triangle;
            TI instance;
            bool valid;
        };

        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT Vec3<T> transform_point(const T transform[12], Vec3<T> point){
            return Vec3<T>{
                transform[0] * point.x + transform[1] * point.y + transform[2]  * point.z + transform[3],
                transform[4] * point.x + transform[5] * point.y + transform[6]  * point.z + transform[7],
                transform[8] * point.x + transform[9] * point.y + transform[10] * point.z + transform[11]
            };
        }

        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT Vec3<T> transform_vector(const T transform[12], Vec3<T> vector){
            return Vec3<T>{
                transform[0] * vector.x + transform[1] * vector.y + transform[2]  * vector.z,
                transform[4] * vector.x + transform[5] * vector.y + transform[6]  * vector.z,
                transform[8] * vector.x + transform[9] * vector.y + transform[10] * vector.z
            };
        }

        // normals transform with the inverse-transpose: multiply by the world_to_object columns
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT Vec3<T> transform_normal(const T world_to_object[12], Vec3<T> normal){
            return Vec3<T>{
                world_to_object[0] * normal.x + world_to_object[4] * normal.y + world_to_object[8]  * normal.z,
                world_to_object[1] * normal.x + world_to_object[5] * normal.y + world_to_object[9]  * normal.z,
                world_to_object[2] * normal.x + world_to_object[6] * normal.y + world_to_object[10] * normal.z
            };
        }

        template <typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void triangle_vertices(const SceneView<T, TI>& scene, TI triangle, Vec3<T>& vertex_a, Vec3<T>& vertex_b, Vec3<T>& vertex_c){
            const MeshView<T, TI>& mesh = scene.meshes[scene.triangle_mesh[triangle]];
            const int* index = &mesh.indices[3 * scene.triangle_local[triangle]];
            vertex_a = to_vec3(&mesh.vertices[3 * (TI)index[0]]);
            vertex_b = to_vec3(&mesh.vertices[3 * (TI)index[1]]);
            vertex_c = to_vec3(&mesh.vertices[3 * (TI)index[2]]);
        }

        // Moeller-Trumbore, no culling; barycentric convention matches the GPU backends
        // (u on the second vertex, v on the third, w0 = 1 - u - v on the first).
        template <typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT bool intersect_triangle(const SceneView<T, TI>& scene, TI triangle, Vec3<T> origin, Vec3<T> direction, T t_min, T t_max, Hit<T, TI>& hit){
            Vec3<T> vertex_a, vertex_b, vertex_c;
            triangle_vertices(scene, triangle, vertex_a, vertex_b, vertex_c);
            const Vec3<T> edge1 = vertex_b - vertex_a;
            const Vec3<T> edge2 = vertex_c - vertex_a;
            const Vec3<T> pvec = cross(direction, edge2);
            const T det = dot(edge1, pvec);
            if(det > (T)-1e-9 && det < (T)1e-9) return false;
            const T inv_det = (T)1 / det;
            const Vec3<T> tvec = origin - vertex_a;
            const T u = dot(tvec, pvec) * inv_det;
            if(u < (T)0 || u > (T)1) return false;
            const Vec3<T> qvec = cross(tvec, edge1);
            const T v = dot(direction, qvec) * inv_det;
            if(v < (T)0 || u + v > (T)1) return false;
            const T t = dot(edge2, qvec) * inv_det;
            if(t <= t_min || t >= t_max) return false;
            hit.t = t;
            hit.u = u;
            hit.v = v;
            hit.triangle = triangle;
            hit.valid = true;
            return true;
        }

        // Branchy slab test: avoids 1/0 = inf, which is undefined behavior under fast-math.
        template <typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT bool intersect_aabb(const BVHNode<T, TI>& node, Vec3<T> origin, Vec3<T> direction, T t_min, T t_max){
            const T origin_array[3] = {origin.x, origin.y, origin.z};
            const T direction_array[3] = {direction.x, direction.y, direction.z};
            for(int axis = 0; axis < 3; axis++){
                const T d = direction_array[axis];
                const T o = origin_array[axis];
                if(d > (T)-1e-12 && d < (T)1e-12){
                    if(o < node.bounds_min[axis] || o > node.bounds_max[axis]) return false;
                }
                else{
                    const T inv = (T)1 / d;
                    T t1 = (node.bounds_min[axis] - o) * inv;
                    T t2 = (node.bounds_max[axis] - o) * inv;
                    if(t1 > t2){ const T tmp = t1; t1 = t2; t2 = tmp; }
                    t_min = maximum(t_min, t1);
                    t_max = minimum(t_max, t2);
                    if(t_min > t_max) return false;
                }
            }
            return true;
        }

        // BLAS traversal against one instance; the ray is transformed into object space with an
        // unnormalized direction, so hit.t stays parameterized in world units and is comparable
        // across instances.
        template <typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void intersect_blas_closest(const SceneView<T, TI>& scene, TI instance_index, Vec3<T> origin, Vec3<T> direction, T t_min, Hit<T, TI>& best){
            const InstanceView<T, TI>& instance = scene.instances[instance_index];
            const ObjectView<T, TI>& object = scene.objects[instance.object];
            if(object.num_nodes == 0) return;
            if(!instance.identity){
                origin = transform_point(instance.world_to_object, origin);
                direction = transform_vector(instance.world_to_object, direction);
            }
            TI stack[constants::TRAVERSAL_STACK_SIZE<TI>];
            TI stack_pointer = 0;
            stack[stack_pointer++] = 0;
            while(stack_pointer > 0){
                const BVHNode<T, TI>& node = object.nodes[stack[--stack_pointer]];
                if(!intersect_aabb(node, origin, direction, t_min, best.t)) continue;
                if(node.count > 0){
                    for(TI i = 0; i < node.count; i++){
                        const TI triangle = object.primitives[node.left_or_first + i];
                        if(intersect_triangle(scene, triangle, origin, direction, t_min, best.t, best)){
                            best.instance = instance_index;
                        }
                    }
                }
                else{
                    if(stack_pointer + 2 <= constants::TRAVERSAL_STACK_SIZE<TI>){
                        stack[stack_pointer++] = node.left_or_first + 1;
                        stack[stack_pointer++] = node.left_or_first;
                    }
                }
            }
        }

        template <typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void traverse_tlas_closest(const SceneView<T, TI>& scene, const BVHNode<T, TI>* tlas_nodes, const TI* tlas_primitives, TI num_tlas_nodes, Vec3<T> origin, Vec3<T> direction, T t_min, Hit<T, TI>& best){
            if(num_tlas_nodes == 0) return;
            TI stack[constants::TRAVERSAL_STACK_SIZE<TI>];
            TI stack_pointer = 0;
            stack[stack_pointer++] = 0;
            while(stack_pointer > 0){
                const BVHNode<T, TI>& node = tlas_nodes[stack[--stack_pointer]];
                if(!intersect_aabb(node, origin, direction, t_min, best.t)) continue;
                if(node.count > 0){
                    for(TI i = 0; i < node.count; i++){
                        const TI instance_index = tlas_primitives[node.left_or_first + i];
                        intersect_blas_closest(scene, instance_index, origin, direction, t_min, best);
                    }
                }
                else{
                    if(stack_pointer + 2 <= constants::TRAVERSAL_STACK_SIZE<TI>){
                        stack[stack_pointer++] = node.left_or_first + 1;
                        stack[stack_pointer++] = node.left_or_first;
                    }
                }
            }
        }

        template <typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT Hit<T, TI> trace_closest(const SceneView<T, TI>& scene, Vec3<T> origin, Vec3<T> direction, T t_min, T t_max){
            Hit<T, TI> best;
            best.t = t_max;
            best.u = 0;
            best.v = 0;
            best.triangle = 0;
            best.instance = 0;
            best.valid = false;
            traverse_tlas_closest(scene, scene.tlas_nodes, scene.tlas_primitives, scene.num_tlas_nodes, origin, direction, t_min, best);
            return best;
        }

        template <typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT bool intersect_blas_any(const SceneView<T, TI>& scene, TI instance_index, Vec3<T> origin, Vec3<T> direction, T t_min, T t_max){
            const InstanceView<T, TI>& instance = scene.instances[instance_index];
            const ObjectView<T, TI>& object = scene.objects[instance.object];
            if(object.num_nodes == 0) return false;
            if(!instance.identity){
                origin = transform_point(instance.world_to_object, origin);
                direction = transform_vector(instance.world_to_object, direction);
            }
            TI stack[constants::TRAVERSAL_STACK_SIZE<TI>];
            TI stack_pointer = 0;
            stack[stack_pointer++] = 0;
            Hit<T, TI> hit;
            hit.valid = false;
            while(stack_pointer > 0){
                const BVHNode<T, TI>& node = object.nodes[stack[--stack_pointer]];
                if(!intersect_aabb(node, origin, direction, t_min, t_max)) continue;
                if(node.count > 0){
                    for(TI i = 0; i < node.count; i++){
                        const TI triangle = object.primitives[node.left_or_first + i];
                        if(intersect_triangle(scene, triangle, origin, direction, t_min, t_max, hit)) return true;
                    }
                }
                else{
                    if(stack_pointer + 2 <= constants::TRAVERSAL_STACK_SIZE<TI>){
                        stack[stack_pointer++] = node.left_or_first + 1;
                        stack[stack_pointer++] = node.left_or_first;
                    }
                }
            }
            return false;
        }

        template <typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT bool traverse_tlas_any(const SceneView<T, TI>& scene, const BVHNode<T, TI>* tlas_nodes, const TI* tlas_primitives, TI num_tlas_nodes, Vec3<T> origin, Vec3<T> direction, T t_min, T t_max){
            if(num_tlas_nodes == 0) return false;
            TI stack[constants::TRAVERSAL_STACK_SIZE<TI>];
            TI stack_pointer = 0;
            stack[stack_pointer++] = 0;
            while(stack_pointer > 0){
                const BVHNode<T, TI>& node = tlas_nodes[stack[--stack_pointer]];
                if(!intersect_aabb(node, origin, direction, t_min, t_max)) continue;
                if(node.count > 0){
                    for(TI i = 0; i < node.count; i++){
                        const TI instance_index = tlas_primitives[node.left_or_first + i];
                        if(intersect_blas_any(scene, instance_index, origin, direction, t_min, t_max)) return true;
                    }
                }
                else{
                    if(stack_pointer + 2 <= constants::TRAVERSAL_STACK_SIZE<TI>){
                        stack[stack_pointer++] = node.left_or_first + 1;
                        stack[stack_pointer++] = node.left_or_first;
                    }
                }
            }
            return false;
        }

        template <typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT bool trace_any(const SceneView<T, TI>& scene, Vec3<T> origin, Vec3<T> direction, T t_min, T t_max){
            return traverse_tlas_any(scene, scene.tlas_nodes, scene.tlas_primitives, scene.num_tlas_nodes, origin, direction, t_min, t_max);
        }

        template <typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT T trace_depth_distance(const SceneView<T, TI>& scene, Vec3<T> origin, Vec3<T> direction, T max_depth){
            const Hit<T, TI> hit = trace_closest(scene, origin, direction, (T)0, max_depth);
            return hit.valid ? hit.t : max_depth;
        }

        template <typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::TI camera_from_pixel(typename SPEC::TI pixel_x, typename SPEC::TI pixel_y){
            return (pixel_y / SPEC::CAM_HEIGHT) * SPEC::GRID_COLS + pixel_x / SPEC::CAM_WIDTH;
        }

        // composed tracing: the shared world plus the overlays attached to the ray's camera.
        // K = 0 compiles down to the raw single-structure trace.
        template <typename SPEC, typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT Hit<T, TI> trace_closest_composed(const SceneView<T, TI>& scene, TI camera, Vec3<T> origin, Vec3<T> direction, T t_min, T t_max){
            Hit<T, TI> best = trace_closest(scene, origin, direction, t_min, t_max);
            if constexpr (SPEC::ENABLE_OVERLAYS){
                for(TI attachment = 0; attachment < SPEC::MAX_OVERLAYS_PER_CAMERA; attachment++){
                    const TI overlay = scene.attachments[camera * SPEC::MAX_OVERLAYS_PER_CAMERA + attachment];
                    if(overlay == ~(TI)0) continue;
                    const OverlayView<T, TI>& view = scene.overlays[overlay];
                    traverse_tlas_closest(scene, view.tlas_nodes, view.tlas_primitives, view.num_tlas_nodes, origin, direction, t_min, best);
                }
            }
            return best;
        }

        template <typename SPEC, typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT bool trace_any_composed(const SceneView<T, TI>& scene, TI camera, Vec3<T> origin, Vec3<T> direction, T t_min, T t_max){
            if(trace_any(scene, origin, direction, t_min, t_max)) return true;
            if constexpr (SPEC::ENABLE_OVERLAYS){
                for(TI attachment = 0; attachment < SPEC::MAX_OVERLAYS_PER_CAMERA; attachment++){
                    const TI overlay = scene.attachments[camera * SPEC::MAX_OVERLAYS_PER_CAMERA + attachment];
                    if(overlay == ~(TI)0) continue;
                    const OverlayView<T, TI>& view = scene.overlays[overlay];
                    if(traverse_tlas_any(scene, view.tlas_nodes, view.tlas_primitives, view.num_tlas_nodes, origin, direction, t_min, t_max)) return true;
                }
            }
            return false;
        }

        template <typename SPEC, typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT T trace_depth_distance_composed(const SceneView<T, TI>& scene, TI camera, Vec3<T> origin, Vec3<T> direction, T max_depth){
            const Hit<T, TI> hit = trace_closest_composed<SPEC>(scene, camera, origin, direction, (T)0, max_depth);
            return hit.valid ? hit.t : max_depth;
        }

        template <typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void expand_triangle_bounds(const SceneView<T, TI>& scene, TI triangle, T bounds_min[3], T bounds_max[3]){
            Vec3<T> vertex_a, vertex_b, vertex_c;
            triangle_vertices(scene, triangle, vertex_a, vertex_b, vertex_c);
            const T vertices[3][3] = {{vertex_a.x, vertex_a.y, vertex_a.z}, {vertex_b.x, vertex_b.y, vertex_b.z}, {vertex_c.x, vertex_c.y, vertex_c.z}};
            for(int vertex_i = 0; vertex_i < 3; vertex_i++){
                for(int axis = 0; axis < 3; axis++){
                    bounds_min[axis] = minimum(bounds_min[axis], vertices[vertex_i][axis]);
                    bounds_max[axis] = maximum(bounds_max[axis], vertices[vertex_i][axis]);
                }
            }
        }

        template <typename TI>
        struct BVHBuildEntry{
            TI node;
            TI start;
            TI count;
        };

        // Median split on the largest centroid-bounds axis; the two-pass partition through
        // temp_primitives preserves relative order, so the build is deterministic. Serves both
        // BLAS (primitives = global triangle ids) and TLAS (primitives = instance indices);
        // bounds/centroid arrays are indexed by primitive value. Returns the node count.
        template <typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT TI build_bvh_nodes(BVHNode<T, TI>* nodes, TI* primitives, TI* temp_primitives, const T* primitive_bounds_min, const T* primitive_bounds_max, const T* centroids, TI count){
            if(count == 0){
                return 0;
            }
            TI node_count = 1;
            BVHBuildEntry<TI> stack[constants::BUILD_STACK_SIZE<TI>];
            TI stack_pointer = 0;
            stack[stack_pointer++] = {0, 0, count};
            while(stack_pointer > 0){
                const BVHBuildEntry<TI> entry = stack[--stack_pointer];
                BVHNode<T, TI>& node = nodes[entry.node];
                for(int axis = 0; axis < 3; axis++){
                    node.bounds_min[axis] = (T)1e30;
                    node.bounds_max[axis] = (T)-1e30;
                }
                T centroid_min[3] = {(T)1e30, (T)1e30, (T)1e30};
                T centroid_max[3] = {(T)-1e30, (T)-1e30, (T)-1e30};
                for(TI i = 0; i < entry.count; i++){
                    const TI primitive = primitives[entry.start + i];
                    for(int axis = 0; axis < 3; axis++){
                        node.bounds_min[axis] = minimum(node.bounds_min[axis], primitive_bounds_min[3 * primitive + axis]);
                        node.bounds_max[axis] = maximum(node.bounds_max[axis], primitive_bounds_max[3 * primitive + axis]);
                        centroid_min[axis] = minimum(centroid_min[axis], centroids[3 * primitive + axis]);
                        centroid_max[axis] = maximum(centroid_max[axis], centroids[3 * primitive + axis]);
                    }
                }
                bool leaf = entry.count <= constants::LEAF_SIZE<TI> || stack_pointer + 2 > constants::BUILD_STACK_SIZE<TI>;
                int split_axis = 0;
                if(!leaf){
                    T max_extent = (T)0;
                    for(int axis = 0; axis < 3; axis++){
                        const T extent = centroid_max[axis] - centroid_min[axis];
                        if(extent > max_extent){
                            max_extent = extent;
                            split_axis = axis;
                        }
                    }
                    if(max_extent <= (T)0) leaf = true;
                }
                if(leaf){
                    node.left_or_first = entry.start;
                    node.count = entry.count;
                    continue;
                }
                const T split = (centroid_min[split_axis] + centroid_max[split_axis]) * (T)0.5;
                TI num_left = 0;
                for(TI i = 0; i < entry.count; i++){
                    const TI primitive = primitives[entry.start + i];
                    if(centroids[3 * primitive + split_axis] < split) temp_primitives[num_left++] = primitive;
                }
                TI num_total = num_left;
                for(TI i = 0; i < entry.count; i++){
                    const TI primitive = primitives[entry.start + i];
                    if(!(centroids[3 * primitive + split_axis] < split)) temp_primitives[num_total++] = primitive;
                }
                if(num_left == 0 || num_left == entry.count){
                    num_left = entry.count / 2; // degenerate split: halve in the current (deterministic) order
                }
                else{
                    for(TI i = 0; i < entry.count; i++){
                        primitives[entry.start + i] = temp_primitives[i];
                    }
                }
                const TI left_child = node_count++;
                const TI right_child = node_count++;
                node.left_or_first = left_child;
                node.count = 0;
                stack[stack_pointer++] = {right_child, entry.start + num_left, entry.count - num_left};
                stack[stack_pointer++] = {left_child, entry.start, num_left};
            }
            return node_count;
        }

        template <typename SPEC, typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT Vec3<T> miss_color(const SceneView<T, TI>& scene, TI pixel_x, TI pixel_y){
            if constexpr (SPEC::SHADING::CHECKER_BACKGROUND){
                const int checker_pattern = ((int)pixel_x / 8) ^ ((int)pixel_y / 8);
                return (checker_pattern & 1) ? to_vec3(scene.miss_color_1) : to_vec3(scene.miss_color_0);
            }
            return to_vec3(scene.miss_color_0);
        }

        template <typename DEVICE, typename SPEC, int DEPTH>
        RL_TOOLS_FUNCTION_PLACEMENT Vec3<typename SPEC::T> trace_rgb(DEVICE& device, const SceneView<typename SPEC::T, typename SPEC::TI>& scene, typename SPEC::TI pixel_x, typename SPEC::TI pixel_y, Vec3<typename SPEC::T> origin, Vec3<typename SPEC::T> direction, typename SPEC::T t_min, typename SPEC::T t_max);

        template <typename DEVICE, typename SPEC, int DEPTH>
        RL_TOOLS_FUNCTION_PLACEMENT Vec3<typename SPEC::T> shade_basic(DEVICE& device, const SceneView<typename SPEC::T, typename SPEC::TI>& scene, typename SPEC::TI pixel_x, typename SPEC::TI pixel_y, Vec3<typename SPEC::T> origin, Vec3<typename SPEC::T> ray_direction, const Hit<typename SPEC::T, typename SPEC::TI>& hit){
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            const auto& math_device = device.math;
            const MeshView<T, TI>& self = scene.meshes[scene.triangle_mesh[hit.triangle]];
            const TI local = scene.triangle_local[hit.triangle];

            Vec3<T> base_color = to_vec3(self.color);
            Vec3<T> normal_geometric = {0, 0, 1};
            Vec3<T> ray_dir = {0, 0, 0};

            if constexpr (SPEC::SHADING::LOAD_TEXTURES || SPEC::SHADING::NORMAL_SHADING || SPEC::SHADING::METALLIC_REFLECTIONS){
                const int* index = &self.indices[3 * local];

                if constexpr (SPEC::SHADING::NORMAL_SHADING || SPEC::SHADING::METALLIC_REFLECTIONS){
                    Vec3<T> vertex_a = to_vec3(&self.vertices[3 * (TI)index[0]]);
                    Vec3<T> vertex_b = to_vec3(&self.vertices[3 * (TI)index[1]]);
                    Vec3<T> vertex_c = to_vec3(&self.vertices[3 * (TI)index[2]]);
                    const InstanceView<T, TI>& instance = scene.instances[hit.instance];
                    if(!instance.identity){
                        vertex_a = transform_point(instance.object_to_world, vertex_a);
                        vertex_b = transform_point(instance.object_to_world, vertex_b);
                        vertex_c = transform_point(instance.object_to_world, vertex_c);
                    }
                    normal_geometric = normalize(math_device, cross(vertex_b - vertex_a, vertex_c - vertex_a));
                    ray_dir = ray_direction;
                }

                if constexpr (SPEC::SHADING::LOAD_TEXTURES){
                    if(self.texture.pixels != nullptr && self.tex_coords != nullptr){
                        const T w0 = (T)1 - hit.u - hit.v;
                        const T tc_x = w0 * self.tex_coords[2 * (TI)index[0] + 0] + hit.u * self.tex_coords[2 * (TI)index[1] + 0] + hit.v * self.tex_coords[2 * (TI)index[2] + 0];
                        const T tc_y = w0 * self.tex_coords[2 * (TI)index[0] + 1] + hit.u * self.tex_coords[2 * (TI)index[1] + 1] + hit.v * self.tex_coords[2 * (TI)index[2] + 1];
                        T tex_color[4];
                        sample_texture(math_device, self.texture, tc_x, tc_y, true, tex_color);
                        base_color = Vec3<T>{tex_color[0], tex_color[1], tex_color[2]} * to_vec3(self.color);
                    }
                }
            }

            Vec3<T> direct = base_color;
            if constexpr (SPEC::SHADING::NORMAL_SHADING){
                const T cosine = dot(ray_dir, normal_geometric);
                direct = ((T)0.2 + (T)0.8 * (cosine < 0 ? -cosine : cosine)) * base_color;
            }

            if constexpr (SPEC::SHADING::METALLIC_REFLECTIONS){
                if constexpr (DEPTH < 1){
                    if(self.metallic > (T)0){
                        const Vec3<T> hit_point = origin + ray_dir * hit.t;
                        const Vec3<T> n = dot(ray_dir, normal_geometric) > (T)0 ? -normal_geometric : normal_geometric;
                        const Vec3<T> reflect_dir = ray_dir - (T)2 * dot(ray_dir, n) * n;

                        const Vec3<T> reflected_color = trace_rgb<DEVICE, SPEC, DEPTH + 1>(device, scene, pixel_x, pixel_y, hit_point, reflect_dir, (T)1e-3, (T)1e20);

                        const T cosine = dot(ray_dir, n);
                        const T cos_theta = cosine < 0 ? -cosine : cosine;
                        const T fresnel = self.metallic * ((T)0.04 + (T)0.96 * math::pow(math_device, (T)1 - cos_theta, (T)5));
                        const Vec3<T> reflect_mix = Vec3<T>{(T)1 - fresnel, (T)1 - fresnel, (T)1 - fresnel} + fresnel * reflected_color;
                        return direct * reflect_mix;
                    }
                }
                return direct;
            }
            return direct;
        }

        template <typename DEVICE, typename SPEC, int DEPTH>
        RL_TOOLS_FUNCTION_PLACEMENT Vec3<typename SPEC::T> shade_pbr(DEVICE& device, const SceneView<typename SPEC::T, typename SPEC::TI>& scene, typename SPEC::TI pixel_x, typename SPEC::TI pixel_y, Vec3<typename SPEC::T> origin, Vec3<typename SPEC::T> ray_dir, const Hit<typename SPEC::T, typename SPEC::TI>& hit){
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            const auto& math_device = device.math;
            const MeshView<T, TI>& self = scene.meshes[scene.triangle_mesh[hit.triangle]];
            const TI local = scene.triangle_local[hit.triangle];
            const int* index = &self.indices[3 * local];

            Vec3<T> vertex_a = to_vec3(&self.vertices[3 * (TI)index[0]]);
            Vec3<T> vertex_b = to_vec3(&self.vertices[3 * (TI)index[1]]);
            Vec3<T> vertex_c = to_vec3(&self.vertices[3 * (TI)index[2]]);
            const InstanceView<T, TI>& hit_instance = scene.instances[hit.instance];
            if(!hit_instance.identity){
                vertex_a = transform_point(hit_instance.object_to_world, vertex_a);
                vertex_b = transform_point(hit_instance.object_to_world, vertex_b);
                vertex_c = transform_point(hit_instance.object_to_world, vertex_c);
            }
            const T w0 = (T)1 - hit.u - hit.v;

            const Vec3<T> edge1 = vertex_b - vertex_a;
            const Vec3<T> edge2 = vertex_c - vertex_a;
            const Vec3<T> normal_geometric = normalize(math_device, cross(edge1, edge2));

            Vec3<T> N;
            if(self.normals != nullptr){
                Vec3<T> normal_interpolated = w0 * to_vec3(&self.normals[3 * (TI)index[0]]) + hit.u * to_vec3(&self.normals[3 * (TI)index[1]]) + hit.v * to_vec3(&self.normals[3 * (TI)index[2]]);
                if(!hit_instance.identity){
                    normal_interpolated = transform_normal(hit_instance.world_to_object, normal_interpolated);
                }
                N = normalize(math_device, normal_interpolated);
            }
            else{
                N = normal_geometric;
            }

            if(dot(ray_dir, N) > (T)0) N = -N;

            T tc_x = 0, tc_y = 0;
            if(self.tex_coords != nullptr){
                tc_x = w0 * self.tex_coords[2 * (TI)index[0] + 0] + hit.u * self.tex_coords[2 * (TI)index[1] + 0] + hit.v * self.tex_coords[2 * (TI)index[2] + 0];
                tc_y = w0 * self.tex_coords[2 * (TI)index[0] + 1] + hit.u * self.tex_coords[2 * (TI)index[1] + 1] + hit.v * self.tex_coords[2 * (TI)index[2] + 1];
            }

            Vec3<T> base_color = to_vec3(self.color);
            T alpha = self.opacity;
            if(self.texture.pixels != nullptr && self.tex_coords != nullptr){
                T tex_color[4];
                sample_texture(math_device, self.texture, tc_x, tc_y, true, tex_color);
                base_color = Vec3<T>{tex_color[0], tex_color[1], tex_color[2]} * to_vec3(self.color);
                alpha *= tex_color[3];
            }

            T metallic = self.metallic;
            T roughness = self.roughness;
            if(self.metallic_roughness_map.pixels != nullptr && self.tex_coords != nullptr){
                T mr_sample[4];
                sample_texture(math_device, self.metallic_roughness_map, tc_x, tc_y, false, mr_sample);
                roughness = mr_sample[1] * self.roughness;
                metallic = mr_sample[2] * self.metallic;
            }

            if(self.normal_map.pixels != nullptr && self.tex_coords != nullptr){
                const T tc0_x = self.tex_coords[2 * (TI)index[0] + 0], tc0_y = self.tex_coords[2 * (TI)index[0] + 1];
                const T tc1_x = self.tex_coords[2 * (TI)index[1] + 0], tc1_y = self.tex_coords[2 * (TI)index[1] + 1];
                const T tc2_x = self.tex_coords[2 * (TI)index[2] + 0], tc2_y = self.tex_coords[2 * (TI)index[2] + 1];
                const T duv1_x = tc1_x - tc0_x, duv1_y = tc1_y - tc0_y;
                const T duv2_x = tc2_x - tc0_x, duv2_y = tc2_y - tc0_y;
                const T det = duv1_x * duv2_y - duv2_x * duv1_y;
                if((det < 0 ? -det : det) > (T)1e-8){
                    const T inv_det = (T)1 / det;
                    Vec3<T> tangent = normalize(math_device, inv_det * (duv2_y * edge1 - duv1_y * edge2));
                    tangent = normalize(math_device, tangent - dot(tangent, N) * N);
                    const Vec3<T> bitangent = cross(N, tangent);
                    T nm_sample[4];
                    sample_texture(math_device, self.normal_map, w0 * tc0_x + hit.u * tc1_x + hit.v * tc2_x, w0 * tc0_y + hit.u * tc1_y + hit.v * tc2_y, false, nm_sample);
                    const Vec3<T> n_tangent = {nm_sample[0] * (T)2 - (T)1, -(nm_sample[1] * (T)2 - (T)1), nm_sample[2] * (T)2 - (T)1};
                    N = normalize(math_device, tangent * n_tangent.x + bitangent * n_tangent.y + N * n_tangent.z);
                }
            }

            roughness = maximum(roughness, (T)0.04);
            const T roughness_alpha = roughness * roughness;
            const T alpha2 = roughness_alpha * roughness_alpha;
            const T k = (roughness + (T)1) * (roughness + (T)1) / (T)8;

            const Vec3<T> V = -ray_dir;
            const T NdotV = maximum(dot(N, V), (T)1e-4);
            const Vec3<T> F0 = Vec3<T>{(T)0.04, (T)0.04, (T)0.04} * ((T)1 - metallic) + base_color * metallic;

            const Vec3<T> hit_point = origin + ray_dir * hit.t;

            Vec3<T> Lo = {0, 0, 0};
            for(TI light_i = 0; light_i < scene.num_lights; light_i++){
                const SceneLight& light = scene.lights[light_i];
                const Vec3<T> Lc = {(T)light.color[0], (T)light.color[1], (T)light.color[2]};
                Vec3<T> L;
                T attenuation = 1;
                T light_distance = (T)1e20;

                if(light.type == 0){
                    L = {(T)light.direction[0], (T)light.direction[1], (T)light.direction[2]};
                }
                else{
                    const Vec3<T> to_light = Vec3<T>{(T)light.position[0], (T)light.position[1], (T)light.position[2]} - hit_point;
                    const T dist = length(math_device, to_light);
                    light_distance = maximum(dist - (T)1e-3, (T)0);
                    L = to_light * ((T)1 / maximum(dist, (T)1e-6));
                    attenuation = (T)1 / ((T)light.attenuation_constant + (T)light.attenuation_linear * dist + (T)light.attenuation_quadratic * dist * dist);
                    if(light.type == 2){
                        const Vec3<T> spot_dir = {(T)light.direction[0], (T)light.direction[1], (T)light.direction[2]};
                        const T cos_angle = dot(-L, spot_dir);
                        const T denom = (T)light.cos_inner_cone - (T)light.cos_outer_cone;
                        const T denom_abs = denom < 0 ? -denom : denom;
                        const T spot_t = (cos_angle - (T)light.cos_outer_cone) / (denom_abs > (T)1e-6 ? denom : (T)1e-6);
                        attenuation *= maximum(minimum(spot_t, (T)1), (T)0);
                    }
                }

                const T NdotL = maximum(dot(N, L), (T)0);
                if(NdotL <= (T)0) continue;
                if constexpr (SPEC::SHADING::PUNCTUAL_LIGHT_SHADOWS){
                    if(light.type != 0 && trace_any_composed<SPEC>(scene, camera_from_pixel<SPEC>(pixel_x, pixel_y), hit_point + N * (T)2e-3, L, (T)1e-3, light_distance)) continue;
                }

                const Vec3<T> H = normalize(math_device, V + L);
                const T NdotH = maximum(dot(N, H), (T)0);
                const T VdotH = maximum(dot(V, H), (T)0);

                const T denom_D = NdotH * NdotH * (alpha2 - (T)1) + (T)1;
                const T D = alpha2 / ((T)3.14159265 * denom_D * denom_D);

                const T G1_V = NdotV / (NdotV * ((T)1 - k) + k);
                const T G1_L = NdotL / (NdotL * ((T)1 - k) + k);
                const T G = G1_V * G1_L;

                const T pow5 = math::pow(math_device, (T)1 - VdotH, (T)5);
                const Vec3<T> F = F0 + (Vec3<T>{1, 1, 1} - F0) * pow5;

                const Vec3<T> specular = F * (D * G * ((T)1 / ((T)4 * NdotV * NdotL + (T)1e-4)));
                const Vec3<T> kd = (Vec3<T>{1, 1, 1} - F) * ((T)1 - metallic);
                const Vec3<T> diffuse = kd * base_color * ((T)1 / (T)3.14159265);

                Lo = Lo + (diffuse + specular) * Lc * (attenuation * NdotL);
            }

            T occlusion = 1;
            if(self.occlusion_map.pixels != nullptr && self.tex_coords != nullptr){
                T ao_sample[4];
                sample_texture(math_device, self.occlusion_map, tc_x, tc_y, false, ao_sample);
                occlusion = ao_sample[0];
            }

            Vec3<T> emissive_color;
            if(self.emissive_map.pixels != nullptr && self.tex_coords != nullptr){
                T em_sample[4];
                sample_texture(math_device, self.emissive_map, tc_x, tc_y, true, em_sample);
                emissive_color = Vec3<T>{em_sample[0], em_sample[1], em_sample[2]} * to_vec3(self.emissive);
            }
            else{
                emissive_color = to_vec3(self.emissive);
            }

            const Vec3<T> ambient = to_vec3(scene.ambient_color) * base_color * (((T)1 - metallic) * occlusion);
            Vec3<T> color = ambient + Lo + emissive_color;

            if constexpr (DEPTH < 1){
                if(metallic > (T)0.1){
                    const Vec3<T> reflect_dir = ray_dir - (T)2 * dot(ray_dir, N) * N;

                    const Vec3<T> reflected_color = trace_rgb<DEVICE, SPEC, DEPTH + 1>(device, scene, pixel_x, pixel_y, hit_point, reflect_dir, (T)1e-3, (T)1e20);

                    const T fresnel_refl = F0.x + ((T)1 - F0.x) * math::pow(math_device, (T)1 - maximum(dot(V, N), (T)0), (T)5);
                    const T reflection_weight = fresnel_refl * ((T)1 - roughness);
                    color = color * ((T)1 - reflection_weight) + reflected_color * reflection_weight;
                }

                const bool transparent = (self.alpha_mode == 2 && alpha < (T)0.99) || (self.alpha_mode == 1 && alpha < self.alpha_cutoff);
                if(transparent){
                    const Vec3<T> behind_color = trace_rgb<DEVICE, SPEC, DEPTH + 1>(device, scene, pixel_x, pixel_y, hit_point, ray_dir, (T)1e-5, (T)1e20);

                    if(self.alpha_mode == 1){
                        color = behind_color;
                    }
                    else{
                        color = color * alpha + behind_color * ((T)1 - alpha);
                    }
                }
            }

            return color;
        }

        template <typename DEVICE, typename SPEC, int DEPTH>
        RL_TOOLS_FUNCTION_PLACEMENT Vec3<typename SPEC::T> trace_rgb(DEVICE& device, const SceneView<typename SPEC::T, typename SPEC::TI>& scene, typename SPEC::TI pixel_x, typename SPEC::TI pixel_y, Vec3<typename SPEC::T> origin, Vec3<typename SPEC::T> direction, typename SPEC::T t_min, typename SPEC::T t_max){
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            const Hit<T, TI> hit = trace_closest_composed<SPEC>(scene, camera_from_pixel<SPEC>(pixel_x, pixel_y), origin, direction, t_min, t_max);
            if(!hit.valid){
                return miss_color<SPEC>(scene, pixel_x, pixel_y);
            }
            if constexpr (SPEC::SHADING::PBR_SHADING){
                return shade_pbr<DEVICE, SPEC, DEPTH>(device, scene, pixel_x, pixel_y, origin, direction, hit);
            }
            else{
                return shade_basic<DEVICE, SPEC, DEPTH>(device, scene, pixel_x, pixel_y, origin, direction, hit);
            }
        }

        struct OutputRGB{};
        struct OutputDepth{};

        template <typename DEVICE, typename SPEC, typename OUTPUT>
        RL_TOOLS_FUNCTION_PLACEMENT void render_frame(DEVICE& device, const SceneView<typename SPEC::T, typename SPEC::TI>& scene){
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            const auto& math_device = device.math;
            constexpr TI MOTION_SAMPLES = SPEC::ENABLE_MOTION_BLUR ? SPEC::MOTION_BLUR_SAMPLES : 1;
            constexpr TI AA_GRID = SPEC::ENABLE_ANTI_ALIASING ? SPEC::ANTI_ALIASING_GRID_SIZE : 1;
            const T inv_aa_grid = (T)1 / (T)AA_GRID;
            for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
                const TI tile_x = (camera_i % SPEC::GRID_COLS) * SPEC::CAM_WIDTH;
                const TI tile_y = (camera_i / SPEC::GRID_COLS) * SPEC::CAM_HEIGHT;
                for(TI y = 0; y < SPEC::CAM_HEIGHT; y++){
                    for(TI x = 0; x < SPEC::CAM_WIDTH; x++){
                        const TI pixel_x = tile_x + x;
                        const TI pixel_y = tile_y + y;
                        const TI fb_offset = camera_i * SPEC::CAM_PIXELS + y * SPEC::CAM_WIDTH + x;
                        Vec3<T> accumulated_rgb = {0, 0, 0};
                        T accumulated_depth = 0;
                        for(TI motion_i = 0; motion_i < MOTION_SAMPLES; motion_i++){
                            Vec3<T> pos, dir_00, dir_du, dir_dv;
                            if constexpr (SPEC::ENABLE_MOTION_BLUR){
                                const Camera<T>& cam_open = scene.cameras_open[camera_i];
                                const Camera<T>& cam_close = scene.cameras_close[camera_i];
                                const T shutter_t = ((T)motion_i + (T)0.5) * ((T)1 / (T)MOTION_SAMPLES);
                                pos = ((T)1 - shutter_t) * to_vec3(cam_open.pos) + shutter_t * to_vec3(cam_close.pos);
                                dir_00 = ((T)1 - shutter_t) * to_vec3(cam_open.dir_00) + shutter_t * to_vec3(cam_close.dir_00);
                                dir_du = ((T)1 - shutter_t) * to_vec3(cam_open.dir_du) + shutter_t * to_vec3(cam_close.dir_du);
                                dir_dv = ((T)1 - shutter_t) * to_vec3(cam_open.dir_dv) + shutter_t * to_vec3(cam_close.dir_dv);
                            }
                            else{
                                const Camera<T>& cam = scene.cameras_close[camera_i];
                                pos = to_vec3(cam.pos);
                                dir_00 = to_vec3(cam.dir_00);
                                dir_du = to_vec3(cam.dir_du);
                                dir_dv = to_vec3(cam.dir_dv);
                            }
                            for(TI aa_y = 0; aa_y < AA_GRID; aa_y++){
                                for(TI aa_x = 0; aa_x < AA_GRID; aa_x++){
                                    const T screen_x = ((T)x + ((T)aa_x + (T)0.5) * inv_aa_grid) / (T)SPEC::CAM_WIDTH;
                                    const T screen_y = ((T)y + ((T)aa_y + (T)0.5) * inv_aa_grid) / (T)SPEC::CAM_HEIGHT;
                                    const Vec3<T> direction = normalize(math_device, dir_00 + screen_x * dir_du + screen_y * dir_dv);
                                    if constexpr (utils::typing::is_same_v<OUTPUT, OutputRGB>){
                                        accumulated_rgb = accumulated_rgb + trace_rgb<DEVICE, SPEC, 0>(device, scene, pixel_x, pixel_y, pos, direction, (T)0, (T)1e30);
                                    }
                                    else{
                                        accumulated_depth += trace_depth_distance_composed<SPEC>(scene, camera_i, pos, direction, scene.max_depth);
                                    }
                                }
                            }
                        }
                        constexpr TI SAMPLES = MOTION_SAMPLES * AA_GRID * AA_GRID;
                        if constexpr (utils::typing::is_same_v<OUTPUT, OutputRGB>){
                            const Vec3<T> color = accumulated_rgb * ((T)1 / (T)SAMPLES);
                            if(scene.observation != nullptr){
                                // the observation is the frame-buffer color before 8-bit
                                // quantization — same transfer curve, full float precision
                                float* observation = scene.observation + fb_offset * 3;
                                if constexpr (SPEC::SHADING::SRGB_OUTPUT){
                                    observation[0] = (float)linear_to_srgb(math_device, clamp01(color.x));
                                    observation[1] = (float)linear_to_srgb(math_device, clamp01(color.y));
                                    observation[2] = (float)linear_to_srgb(math_device, clamp01(color.z));
                                }
                                else{
                                    observation[0] = (float)clamp01(color.x);
                                    observation[1] = (float)clamp01(color.y);
                                    observation[2] = (float)clamp01(color.z);
                                }
                            }
                            if constexpr (SPEC::SHADING::SRGB_OUTPUT){
                                scene.frame_buffer[fb_offset] = make_srgb_rgba_from_linear(math_device, color);
                            }
                            else{
                                scene.frame_buffer[fb_offset] = make_linear_rgba_from_linear(color);
                            }
                        }
                        else{
                            scene.depth_buffer[fb_offset] = (float)(accumulated_depth * ((T)1 / (T)SAMPLES));
                        }
                    }
                }
            }
        }

        // one dynamic-motion-blur pass: camera lerped at the caller's shutter time, overlay
        // geometry already rebuilt for this sample by the caller; adds this pass's linear mean
        // into the accumulation buffer. Each iteration owns its pixel — no atomics.
        template <typename DEVICE, typename SPEC, typename OUTPUT>
        RL_TOOLS_FUNCTION_PLACEMENT void render_frame_accumulate(DEVICE& device, const SceneView<typename SPEC::T, typename SPEC::TI>& scene, typename SPEC::T shutter_t){
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            const auto& math_device = device.math;
            constexpr TI AA_GRID = SPEC::ENABLE_ANTI_ALIASING ? SPEC::ANTI_ALIASING_GRID_SIZE : 1;
            const T inv_aa_grid = (T)1 / (T)AA_GRID;
            for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
                const Camera<T>& cam_open = scene.cameras_open[camera_i];
                const Camera<T>& cam_close = scene.cameras_close[camera_i];
                const Vec3<T> pos = ((T)1 - shutter_t) * to_vec3(cam_open.pos) + shutter_t * to_vec3(cam_close.pos);
                const Vec3<T> dir_00 = ((T)1 - shutter_t) * to_vec3(cam_open.dir_00) + shutter_t * to_vec3(cam_close.dir_00);
                const Vec3<T> dir_du = ((T)1 - shutter_t) * to_vec3(cam_open.dir_du) + shutter_t * to_vec3(cam_close.dir_du);
                const Vec3<T> dir_dv = ((T)1 - shutter_t) * to_vec3(cam_open.dir_dv) + shutter_t * to_vec3(cam_close.dir_dv);
                const TI tile_x = (camera_i % SPEC::GRID_COLS) * SPEC::CAM_WIDTH;
                const TI tile_y = (camera_i / SPEC::GRID_COLS) * SPEC::CAM_HEIGHT;
                for(TI y = 0; y < SPEC::CAM_HEIGHT; y++){
                    for(TI x = 0; x < SPEC::CAM_WIDTH; x++){
                        const TI pixel_x = tile_x + x;
                        const TI pixel_y = tile_y + y;
                        const TI fb_offset = camera_i * SPEC::CAM_PIXELS + y * SPEC::CAM_WIDTH + x;
                        Vec3<T> accumulated_rgb = {0, 0, 0};
                        T accumulated_depth = 0;
                        for(TI aa_y = 0; aa_y < AA_GRID; aa_y++){
                            for(TI aa_x = 0; aa_x < AA_GRID; aa_x++){
                                const T screen_x = ((T)x + ((T)aa_x + (T)0.5) * inv_aa_grid) / (T)SPEC::CAM_WIDTH;
                                const T screen_y = ((T)y + ((T)aa_y + (T)0.5) * inv_aa_grid) / (T)SPEC::CAM_HEIGHT;
                                const Vec3<T> direction = normalize(math_device, dir_00 + screen_x * dir_du + screen_y * dir_dv);
                                if constexpr (utils::typing::is_same_v<OUTPUT, OutputRGB>){
                                    accumulated_rgb = accumulated_rgb + trace_rgb<DEVICE, SPEC, 0>(device, scene, pixel_x, pixel_y, pos, direction, (T)0, (T)1e30);
                                }
                                else{
                                    accumulated_depth += trace_depth_distance_composed<SPEC>(scene, camera_i, pos, direction, scene.max_depth);
                                }
                            }
                        }
                        constexpr TI SAMPLES = AA_GRID * AA_GRID;
                        if constexpr (utils::typing::is_same_v<OUTPUT, OutputRGB>){
                            const Vec3<T> color = accumulated_rgb * ((T)1 / (T)SAMPLES);
                            scene.rgb_accumulation[fb_offset * 3 + 0] += (float)color.x;
                            scene.rgb_accumulation[fb_offset * 3 + 1] += (float)color.y;
                            scene.rgb_accumulation[fb_offset * 3 + 2] += (float)color.z;
                        }
                        else{
                            scene.depth_accumulation[fb_offset] += (float)(accumulated_depth * ((T)1 / (T)SAMPLES));
                        }
                    }
                }
            }
        }

        // divides the accumulated linear radiance by the pass count and writes the declared
        // outputs with the same transfer curve + quantization as the single-launch path
        template <typename DEVICE, typename SPEC, typename OUTPUT>
        RL_TOOLS_FUNCTION_PLACEMENT void resolve_frame(DEVICE& device, const SceneView<typename SPEC::T, typename SPEC::TI>& scene){
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            const auto& math_device = device.math;
            const T inv_samples = (T)1 / (T)SPEC::MOTION_BLUR_SAMPLES;
            for(TI fb_offset = 0; fb_offset < SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS; fb_offset++){
                if constexpr (utils::typing::is_same_v<OUTPUT, OutputRGB>){
                    const Vec3<T> color = {
                        (T)scene.rgb_accumulation[fb_offset * 3 + 0] * inv_samples,
                        (T)scene.rgb_accumulation[fb_offset * 3 + 1] * inv_samples,
                        (T)scene.rgb_accumulation[fb_offset * 3 + 2] * inv_samples
                    };
                    if(scene.observation != nullptr){
                        float* observation = scene.observation + fb_offset * 3;
                        if constexpr (SPEC::SHADING::SRGB_OUTPUT){
                            observation[0] = (float)linear_to_srgb(math_device, clamp01(color.x));
                            observation[1] = (float)linear_to_srgb(math_device, clamp01(color.y));
                            observation[2] = (float)linear_to_srgb(math_device, clamp01(color.z));
                        }
                        else{
                            observation[0] = (float)clamp01(color.x);
                            observation[1] = (float)clamp01(color.y);
                            observation[2] = (float)clamp01(color.z);
                        }
                    }
                    if constexpr (SPEC::SHADING::SRGB_OUTPUT){
                        scene.frame_buffer[fb_offset] = make_srgb_rgba_from_linear(math_device, color);
                    }
                    else{
                        scene.frame_buffer[fb_offset] = make_linear_rgba_from_linear(color);
                    }
                }
                else{
                    scene.depth_buffer[fb_offset] = (float)((T)scene.depth_accumulation[fb_offset] * inv_samples);
                }
            }
        }

        // single-sample by design: instance labels cannot be averaged, so anti-aliasing and
        // motion blur do not apply (shutter-close camera, pixel-center ray)
        template <typename DEVICE, typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void render_segmentation_frame(DEVICE& device, const SceneView<typename SPEC::T, typename SPEC::TI>& scene){
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            const auto& math_device = device.math;
            for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
                const Camera<T>& cam = scene.cameras_close[camera_i];
                const Vec3<T> pos = to_vec3(cam.pos);
                const Vec3<T> dir_00 = to_vec3(cam.dir_00);
                const Vec3<T> dir_du = to_vec3(cam.dir_du);
                const Vec3<T> dir_dv = to_vec3(cam.dir_dv);
                for(TI y = 0; y < SPEC::CAM_HEIGHT; y++){
                    for(TI x = 0; x < SPEC::CAM_WIDTH; x++){
                        const TI fb_offset = camera_i * SPEC::CAM_PIXELS + y * SPEC::CAM_WIDTH + x;
                        const T screen_x = ((T)x + (T)0.5) / (T)SPEC::CAM_WIDTH;
                        const T screen_y = ((T)y + (T)0.5) / (T)SPEC::CAM_HEIGHT;
                        const Vec3<T> direction = normalize(math_device, dir_00 + screen_x * dir_du + screen_y * dir_dv);
                        const Hit<T, TI> hit = trace_closest_composed<SPEC>(scene, camera_i, pos, direction, (T)0, (T)1e30);
                        scene.segmentation_buffer[fb_offset] = hit.valid ? (SPEC::SEMANTIC_SEGMENTATION ? scene.instance_classes[hit.instance] : (unsigned int)hit.instance) : 0xFFFFFFFFu;
                    }
                }
            }
        }

        // world point → screen fraction through the linear camera model: solves
        // dir_00 + screen_x·dir_du + screen_y·dir_dv = lambda·(point − pos) by Cramer's rule;
        // returns false for a degenerate basis or a point at/behind the camera (lambda <= 0)
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT bool project_camera(const Camera<T>& cam, Vec3<T> point, T& screen_x, T& screen_y){
            const Vec3<T> direction = point - to_vec3(cam.pos);
            const Vec3<T> dir_du = to_vec3(cam.dir_du);
            const Vec3<T> dir_dv = to_vec3(cam.dir_dv);
            const Vec3<T> negative_direction = -direction;
            const Vec3<T> cross_dv_negative_direction = cross(dir_dv, negative_direction);
            const T det = dot(dir_du, cross_dv_negative_direction);
            if(det > (T)-1e-12 && det < (T)1e-12){
                return false;
            }
            const T inv_det = (T)1 / det;
            const Vec3<T> b = -to_vec3(cam.dir_00);
            screen_x = dot(b, cross_dv_negative_direction) * inv_det;
            screen_y = dot(dir_du, cross(b, negative_direction)) * inv_det;
            const T lambda = dot(dir_du, cross(dir_dv, b)) * inv_det;
            return lambda > (T)0;
        }

        // single-sample by design (shutter-close camera, pixel-center ray): backward flow of the
        // shutter-close frame in pixels — the hit point is carried to shutter open by its
        // instance's shutter delta (identity for the static world) and projected through the
        // shutter-open camera; miss (and behind-the-open-camera projections) write (0, 0)
        template <typename DEVICE, typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void render_flow_frame(DEVICE& device, const SceneView<typename SPEC::T, typename SPEC::TI>& scene){
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            const auto& math_device = device.math;
            for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
                const Camera<T>& cam = scene.cameras_close[camera_i];
                const Camera<T>& cam_open = scene.cameras_open[camera_i];
                const Vec3<T> pos = to_vec3(cam.pos);
                const Vec3<T> dir_00 = to_vec3(cam.dir_00);
                const Vec3<T> dir_du = to_vec3(cam.dir_du);
                const Vec3<T> dir_dv = to_vec3(cam.dir_dv);
                for(TI y = 0; y < SPEC::CAM_HEIGHT; y++){
                    for(TI x = 0; x < SPEC::CAM_WIDTH; x++){
                        const TI fb_offset = camera_i * SPEC::CAM_PIXELS + y * SPEC::CAM_WIDTH + x;
                        const T screen_x = ((T)x + (T)0.5) / (T)SPEC::CAM_WIDTH;
                        const T screen_y = ((T)y + (T)0.5) / (T)SPEC::CAM_HEIGHT;
                        const Vec3<T> direction = normalize(math_device, dir_00 + screen_x * dir_du + screen_y * dir_dv);
                        const Hit<T, TI> hit = trace_closest_composed<SPEC>(scene, camera_i, pos, direction, (T)0, (T)1e30);
                        T flow_u = 0, flow_v = 0;
                        if(hit.valid){
                            Vec3<T> point = pos + direction * hit.t;
                            if constexpr (SPEC::ENABLE_OVERLAYS){
                                if(hit.instance >= scene.first_overlay_instance){
                                    point = transform_point(&scene.flow_deltas[(TI)(hit.instance - scene.first_overlay_instance) * 12], point);
                                }
                            }
                            T open_x, open_y;
                            if(project_camera(cam_open, point, open_x, open_y)){
                                flow_u = ((T)x + (T)0.5) - open_x * (T)SPEC::CAM_WIDTH;
                                flow_v = ((T)y + (T)0.5) - open_y * (T)SPEC::CAM_HEIGHT;
                            }
                        }
                        scene.flow_buffer[fb_offset * 2 + 0] = (float)flow_u;
                        scene.flow_buffer[fb_offset * 2 + 1] = (float)flow_v;
                    }
                }
            }
        }

        // single-sample by design: unit normals cannot be averaged, so anti-aliasing and motion
        // blur do not apply (shutter-close camera, pixel-center ray); the geometric normal is
        // world-frame (FLU), oriented against the ray, zero on miss
        template <typename DEVICE, typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void render_normals_frame(DEVICE& device, const SceneView<typename SPEC::T, typename SPEC::TI>& scene){
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            const auto& math_device = device.math;
            for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
                const Camera<T>& cam = scene.cameras_close[camera_i];
                const Vec3<T> pos = to_vec3(cam.pos);
                const Vec3<T> dir_00 = to_vec3(cam.dir_00);
                const Vec3<T> dir_du = to_vec3(cam.dir_du);
                const Vec3<T> dir_dv = to_vec3(cam.dir_dv);
                for(TI y = 0; y < SPEC::CAM_HEIGHT; y++){
                    for(TI x = 0; x < SPEC::CAM_WIDTH; x++){
                        const TI fb_offset = camera_i * SPEC::CAM_PIXELS + y * SPEC::CAM_WIDTH + x;
                        const T screen_x = ((T)x + (T)0.5) / (T)SPEC::CAM_WIDTH;
                        const T screen_y = ((T)y + (T)0.5) / (T)SPEC::CAM_HEIGHT;
                        const Vec3<T> direction = normalize(math_device, dir_00 + screen_x * dir_du + screen_y * dir_dv);
                        const Hit<T, TI> hit = trace_closest_composed<SPEC>(scene, camera_i, pos, direction, (T)0, (T)1e30);
                        Vec3<T> normal = {0, 0, 0};
                        if(hit.valid){
                            Vec3<T> vertex_a, vertex_b, vertex_c;
                            triangle_vertices(scene, hit.triangle, vertex_a, vertex_b, vertex_c);
                            const InstanceView<T, TI>& instance = scene.instances[hit.instance];
                            if(!instance.identity){
                                vertex_a = transform_point(instance.object_to_world, vertex_a);
                                vertex_b = transform_point(instance.object_to_world, vertex_b);
                                vertex_c = transform_point(instance.object_to_world, vertex_c);
                            }
                            normal = normalize(math_device, cross(vertex_b - vertex_a, vertex_c - vertex_a));
                            if(dot(direction, normal) > (T)0) normal = -normal;
                        }
                        scene.normals_buffer[fb_offset * 3 + 0] = (float)normal.x;
                        scene.normals_buffer[fb_offset * 3 + 1] = (float)normal.y;
                        scene.normals_buffer[fb_offset * 3 + 2] = (float)normal.z;
                    }
                }
            }
        }

        template <typename DEVICE, typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void render_collision(DEVICE& device, const SceneView<typename SPEC::T, typename SPEC::TI>& scene){
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            const auto& math_device = device.math;
            for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
                const Camera<T>& cam = scene.cameras_close[camera_i];
                for(TI probe_i = 0; probe_i < SPEC::NUM_PROBES; probe_i++){
                    Vec3<T> direction;
                    if(probe_i == 0){
                        direction = normalize(math_device, to_vec3(cam.dir_00) + (T)0.5 * to_vec3(cam.dir_du) + (T)0.5 * to_vec3(cam.dir_dv));
                    }
                    else{
                        direction = to_vec3(&scene.probe_directions[3 * probe_i]);
                    }
                    const Hit<T, TI> hit = trace_closest_composed<SPEC>(scene, camera_i, to_vec3(cam.pos), direction, (T)1e-3, scene.max_dist);
                    CollisionResult result;
                    if(!hit.valid){
                        result.distance = (float)scene.max_dist;
                        result.hit = 0;
                    }
                    else{
                        result.distance = (float)hit.t;
                        result.hit = 1;
                    }
                    scene.collision_results[camera_i * SPEC::NUM_PROBES + probe_i] = result;
                }
            }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif

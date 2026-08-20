#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_WEBGPU_BVH_SAH_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_WEBGPU_BVH_SAH_H

// Deterministic binned-SAH BVH builder: 16 bins on the largest centroid axis, stable
// order-preserving partition (fixed bin count, min/max-only bounds accumulation — no
// order-dependent float sums), drop-in signature-compatible with generic::build_bvh_nodes.
// The generic median-split builder remains the baseline (RL_TOOLS_WEBGPU_BVH=median) and is
// still used for the per-frame overlay TLAS rebuilds.
#include "../generic/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::raytracing::backends::webgpu{
    namespace sah{
        template <typename TI> constexpr TI NUM_BINS = 16;
        template <typename TI> constexpr TI MAX_LEAF_SIZE = 4;

        template <typename T>
        T half_area(const T bounds_min[3], const T bounds_max[3]){
            const T dx = bounds_max[0] - bounds_min[0];
            const T dy = bounds_max[1] - bounds_min[1];
            const T dz = bounds_max[2] - bounds_min[2];
            return dx * dy + dy * dz + dz * dx;
        }
    }

    template <typename T, typename TI>
    TI build_bvh_nodes_sah(generic::BVHNode<T, TI>* nodes, TI* primitives, TI* temp_primitives, const T* primitive_bounds_min, const T* primitive_bounds_max, const T* centroids, TI count){
        namespace constants = generic::constants;
        if(count == 0){
            return 0;
        }
        TI node_count = 1;
        generic::BVHBuildEntry<TI> stack[constants::BUILD_STACK_SIZE<TI>];
        TI stack_pointer = 0;
        stack[stack_pointer++] = {0, 0, count};
        while(stack_pointer > 0){
            const generic::BVHBuildEntry<TI> entry = stack[--stack_pointer];
            generic::BVHNode<T, TI>& node = nodes[entry.node];
            for(int axis = 0; axis < 3; axis++){
                node.bounds_min[axis] = (T)1e30;
                node.bounds_max[axis] = (T)-1e30;
            }
            T centroid_min[3] = {(T)1e30, (T)1e30, (T)1e30};
            T centroid_max[3] = {(T)-1e30, (T)-1e30, (T)-1e30};
            for(TI i = 0; i < entry.count; i++){
                const TI primitive = primitives[entry.start + i];
                for(int axis = 0; axis < 3; axis++){
                    node.bounds_min[axis] = generic::minimum(node.bounds_min[axis], primitive_bounds_min[3 * primitive + axis]);
                    node.bounds_max[axis] = generic::maximum(node.bounds_max[axis], primitive_bounds_max[3 * primitive + axis]);
                    centroid_min[axis] = generic::minimum(centroid_min[axis], centroids[3 * primitive + axis]);
                    centroid_max[axis] = generic::maximum(centroid_max[axis], centroids[3 * primitive + axis]);
                }
            }
            bool leaf = entry.count <= 1 || stack_pointer + 2 > constants::BUILD_STACK_SIZE<TI>;
            int split_axis = 0;
            T max_extent = (T)0;
            if(!leaf){
                for(int axis = 0; axis < 3; axis++){
                    const T extent = centroid_max[axis] - centroid_min[axis];
                    if(extent > max_extent){
                        max_extent = extent;
                        split_axis = axis;
                    }
                }
                if(max_extent <= (T)0 && entry.count <= sah::MAX_LEAF_SIZE<TI>){
                    leaf = true;
                }
            }
            TI num_left = 0;
            if(!leaf && max_extent > (T)0){
                constexpr TI BINS = sah::NUM_BINS<TI>;
                TI bin_counts[BINS] = {};
                T bin_bounds_min[BINS][3];
                T bin_bounds_max[BINS][3];
                for(TI bin = 0; bin < BINS; bin++){
                    for(int axis = 0; axis < 3; axis++){
                        bin_bounds_min[bin][axis] = (T)1e30;
                        bin_bounds_max[bin][axis] = (T)-1e30;
                    }
                }
                const T bin_scale = (T)BINS / max_extent;
                for(TI i = 0; i < entry.count; i++){
                    const TI primitive = primitives[entry.start + i];
                    TI bin = (TI)((centroids[3 * primitive + split_axis] - centroid_min[split_axis]) * bin_scale);
                    bin = bin < BINS ? bin : BINS - 1;
                    bin_counts[bin]++;
                    for(int axis = 0; axis < 3; axis++){
                        bin_bounds_min[bin][axis] = generic::minimum(bin_bounds_min[bin][axis], primitive_bounds_min[3 * primitive + axis]);
                        bin_bounds_max[bin][axis] = generic::maximum(bin_bounds_max[bin][axis], primitive_bounds_max[3 * primitive + axis]);
                    }
                }
                // sweep the BINS - 1 split planes: prefix areas left-to-right, suffix right-to-left
                T left_area[BINS - 1];
                TI left_count[BINS - 1];
                {
                    T sweep_min[3] = {(T)1e30, (T)1e30, (T)1e30};
                    T sweep_max[3] = {(T)-1e30, (T)-1e30, (T)-1e30};
                    TI sweep_count = 0;
                    for(TI bin = 0; bin < BINS - 1; bin++){
                        sweep_count += bin_counts[bin];
                        for(int axis = 0; axis < 3; axis++){
                            sweep_min[axis] = generic::minimum(sweep_min[axis], bin_bounds_min[bin][axis]);
                            sweep_max[axis] = generic::maximum(sweep_max[axis], bin_bounds_max[bin][axis]);
                        }
                        left_area[bin] = sah::half_area(sweep_min, sweep_max);
                        left_count[bin] = sweep_count;
                    }
                }
                TI best_plane = 0;
                T best_cost = (T)1e30;
                {
                    T sweep_min[3] = {(T)1e30, (T)1e30, (T)1e30};
                    T sweep_max[3] = {(T)-1e30, (T)-1e30, (T)-1e30};
                    TI sweep_count = 0;
                    for(TI plane = BINS - 1; plane > 0; plane--){
                        sweep_count += bin_counts[plane];
                        for(int axis = 0; axis < 3; axis++){
                            sweep_min[axis] = generic::minimum(sweep_min[axis], bin_bounds_min[plane][axis]);
                            sweep_max[axis] = generic::maximum(sweep_max[axis], bin_bounds_max[plane][axis]);
                        }
                        const TI num_l = left_count[plane - 1];
                        const TI num_r = sweep_count;
                        if(num_l == 0 || num_r == 0){
                            continue;
                        }
                        const T cost = left_area[plane - 1] * (T)num_l + sah::half_area(sweep_min, sweep_max) * (T)num_r;
                        if(cost < best_cost){
                            best_cost = cost;
                            best_plane = plane;
                        }
                    }
                }
                const T parent_area = sah::half_area(node.bounds_min, node.bounds_max);
                const bool split_worthwhile = best_plane > 0
                    && (entry.count > sah::MAX_LEAF_SIZE<TI> || (parent_area > (T)0 && (T)1 + best_cost / parent_area < (T)entry.count));
                if(split_worthwhile){
                    // stable order-preserving partition through temp_primitives, like the generic builder
                    for(TI i = 0; i < entry.count; i++){
                        const TI primitive = primitives[entry.start + i];
                        TI bin = (TI)((centroids[3 * primitive + split_axis] - centroid_min[split_axis]) * bin_scale);
                        bin = bin < BINS ? bin : BINS - 1;
                        if(bin < best_plane) temp_primitives[num_left++] = primitive;
                    }
                    TI num_total = num_left;
                    for(TI i = 0; i < entry.count; i++){
                        const TI primitive = primitives[entry.start + i];
                        TI bin = (TI)((centroids[3 * primitive + split_axis] - centroid_min[split_axis]) * bin_scale);
                        bin = bin < BINS ? bin : BINS - 1;
                        if(!(bin < best_plane)) temp_primitives[num_total++] = primitive;
                    }
                    for(TI i = 0; i < entry.count; i++){
                        primitives[entry.start + i] = temp_primitives[i];
                    }
                }
            }
            if(num_left == 0){
                // no worthwhile SAH split: leaf when small enough, deterministic halving otherwise
                // (zero centroid extent or a degenerate binning cannot be split spatially)
                if(entry.count <= sah::MAX_LEAF_SIZE<TI> || stack_pointer + 2 > constants::BUILD_STACK_SIZE<TI>){
                    node.left_or_first = entry.start;
                    node.count = entry.count;
                    continue;
                }
                num_left = entry.count / 2;
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

}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif

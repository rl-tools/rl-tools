#ifndef QUATERNION_HELPER_H
#define QUATERNION_HELPER_H
#include "general_helper.h"

#ifndef BACKPROP_TOOLS_FUNCTION_PLACEMENT
#define BACKPROP_TOOLS_FUNCTION_PLACEMENT
#endif

template <typename T>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void quaternion_derivative(const T q[4], const T omega[3], T q_dot[4]) {
    // FLOPS: 3 (MAC) * 4 + 4 = 16
    q_dot[0] = -q[1]*omega[0] - q[2]*omega[1] - q[3]*omega[2];
    q_dot[1] =  q[0]*omega[0] + q[2]*omega[2] - q[3]*omega[1];
    q_dot[2] =  q[0]*omega[1] + q[3]*omega[0] - q[1]*omega[2];
    q_dot[3] =  q[0]*omega[2] + q[1]*omega[1] - q[2]*omega[0];
    scalar_multiply<T, 4>(q_dot, 0.5);
}


template <typename T>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void rotate_vector_by_quaternion(const T q[4], const T v[3], T v_out[3]) {
//    v_out[0] = q[0]*(q[0]*v[0] + q[2]*v[2] - q[3]*v[1]) + q[2]*(q[1]*v[1] + q[0]*v[2] - q[2]*v[0]) - q[3]*(q[0]*v[1] + q[3]*v[0] - q[1]*v[2]) - q[1]*(-q[1]*v[0] - q[2]*v[1] - q[3]*v[2]);
//    v_out[1] = q[0]*(q[0]*v[1] + q[3]*v[0] - q[1]*v[2]) + q[3]*(q[0]*v[0] + q[2]*v[2] - q[3]*v[1]) - q[1]*(q[1]*v[1] + q[0]*v[2] - q[2]*v[0]) - q[2]*(-q[1]*v[0] - q[2]*v[1] - q[3]*v[2]);
//    v_out[2] = q[0]*(q[1]*v[1] + q[0]*v[2] - q[2]*v[0]) + q[1]*(q[0]*v[1] + q[3]*v[0] - q[1]*v[2]) - q[2]*(q[0]*v[0] + q[2]*v[2] - q[3]*v[1]) - q[3]*(-q[1]*v[0] - q[2]*v[1] - q[3]*v[2]);
    // FLOPS: 6 + 3 + 6 + 3 + 3 = 21

    T var[3];
    cross_product<T>(&q[1], v, var); // 6 flops
    scalar_multiply<T, 3>(var, 2); // 3 flops
    cross_product<T>(&q[1], var, v_out); // 6 flops
    scalar_multiply_accumulate<T, 3>(var, q[0], v_out); // 3 flops
    vector_add_accumulate<T, 3>(v, v_out); // 3 flops
}


#endif
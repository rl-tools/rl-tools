#ifndef GENERAL_HELPER_H
#define GENERAL_HELPER_H

#ifndef BACKPROP_TOOLS_FUNCTION_PLACEMENT
#define BACKPROP_TOOLS_FUNCTION_PLACEMENT
#endif

template <typename T, int N>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void scalar_multiply(const T v[N], const T s, T out[N]) {
    for(int i = 0; i < N; i++) {
        out[i] = v[i]*s;
    }
}

template <typename T, int N>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void scalar_multiply(T v[N], const T s) {
    for(int i = 0; i < N; i++) {
        v[i] *= s;
    }
}

template <typename T, int N>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void scalar_multiply_accumulate(const T v[N], T s, T out[N]) {
    for(int i = 0; i < N; i++) {
        out[i] += v[i]*s;
    }
}

template <typename T, int M, int N>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void matrix_vector_product(const T A[M][N], const T v[N], T out[M]) {
    for(int i = 0; i < M; i++) {
        out[i] = 0;
        for(int j = 0; j < N; j++) {
            out[i] += A[i][j]*v[j];
        }
    }
}

template <typename T>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void cross_product(const T v1[3], const T v2[3], T out[3]) {
    // flops: 2 * 3 = 6
    out[0] = v1[1]*v2[2] - v1[2]*v2[1];
    out[1] = v1[2]*v2[0] - v1[0]*v2[2];
    out[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

template <typename T>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void cross_product_accumulate(const T v1[3], const T v2[3], T out[3]) {
    out[0] += v1[1]*v2[2] - v1[2]*v2[1];
    out[1] += v1[2]*v2[0] - v1[0]*v2[2];
    out[2] += v1[0]*v2[1] - v1[1]*v2[0];
}

template <typename T, int N>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void vector_add(const T v1[N], const T v2[N], T out[N]) {
    for(int i = 0; i < N; i++) {
        out[i] = v1[i] + v2[i];
    }
}
template <typename T, int N>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void vector_add_accumulate(const T v1[N], const T v2[N], T out[N]) {
    for(int i = 0; i < N; i++) {
        out[i] += v1[i] + v2[i];
    }
}
template <typename T, int N>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void vector_add_accumulate(T const v[N], T out[N]) {
    for(int i = 0; i < N; i++) {
        out[i] += v[i];
    }
}

template <typename T, int N>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void vector_sub(T const v1[N], const T v2[N], T out[N]) {
    for(int i = 0; i < N; i++) {
        out[i] = v1[i] - v2[i];
    }
}
template <typename T, int N>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void vector_sub_accumulate(const T v1[N], const T v2[N], T out[N]) {
    for(int i = 0; i < N; i++) {
        out[i] += v1[i] - v2[i];
    }
}
template <typename T, int N>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void vector_sub_accumulate(const T v[N], T out[N]) {
    for(int i = 0; i < N; i++) {
        out[i] -= v[i];
    }
}

template <typename T, int N>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void fill(T v[N], T s) {
    for(int i = 0; i < N; i++) {
        v[i] = s;
    }
}

template <typename T, int N>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void assign(const T source[N], T target[N]) {
    for(int i = 0; i < N; i++) {
        target[i] = source[i];
    }
}

template <typename T, int M, int N>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void assign(const T source[M][N], T target[M][N]) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            target[i][j] = source[i][j];
        }
    }
}

template <typename T, int M, int N, int P>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void assign(const T source[M][N][P], T target[M][N][P]) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            for(int k = 0; k < P; k++) {
                target[i][j][k] = source[i][j][k];
            }
        }
    }
}
template <typename T, int M, int N, int P>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void assign(const T source[P], T target[M][N][P]) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            for(int k = 0; k < P; k++) {
                target[i][j][k] = source[k];
            }
        }
    }
}

template <typename T, int N>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void normalize(const T source[N], T target[N]) {
    T acc = 0;
    for(int i = 0; i < N; i++) {
        acc += source[i]*source[i];
    }
    acc = sqrt(acc);
    for(int i = 0; i < N; i++) {
        target[i] = source[i]/acc;
    }
}

#endif
#ifndef YAW_PREDICTOR_INFERENCE_H
#define YAW_PREDICTOR_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct YawPredictorHandle YawPredictorHandle;

// Load model from HDF5 checkpoint. Returns NULL on failure.
YawPredictorHandle* yaw_predictor_create(const char* h5_path);

// Run inference on two 64x64x3 NHWC float images (values in [0,1]).
// Returns the normalized displacement (delta_yaw / half_hfov) in [-1, 1].
float yaw_predictor_evaluate(YawPredictorHandle* handle,
                             const float* image_a,
                             const float* image_b);

void yaw_predictor_destroy(YawPredictorHandle* handle);

#ifdef __cplusplus
}
#endif

#endif

#ifndef LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_ARM_H
#define LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_ARM_H

#include "operations_generic.h"
//#include <layer_in_c/utils/generic/memcpy.h>
#include <layer_in_c/devices/arm.h>
#include "arm_math.h"

namespace layer_in_c{
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    void evaluate(devices::ARM<DEV_SPEC>& device, const nn::layers::dense::Layer<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output) {
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        static_assert(INPUT_SPEC::ROW_PITCH == INPUT_SPEC::COLS);
        static_assert(INPUT_SPEC::COL_PITCH == 1);
        static_assert(OUTPUT_SPEC::ROW_PITCH == OUTPUT_SPEC::COLS);
        static_assert(OUTPUT_SPEC::COL_PITCH == 1);
        static_assert(decltype(layer.weights)::ROW_PITCH == INPUT_SPEC::COLS);
        static_assert(decltype(layer.weights)::COL_PITCH == 1);
        static_assert(decltype(layer.biases)::COL_PITCH == 1);
        static_assert(decltype(layer.biases)::ROW_PITCH == decltype(layer.biases)::COLS);
        static_assert(utils::typing::is_same_v<typename LAYER_SPEC::T, float>);

        // Warning do not use the same buffer for input and output!
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        static_assert(BATCH_SIZE == 1);
        using DEVICE = devices::ARM<DEV_SPEC>;
        using T = typename LAYER_SPEC::T;
        using TI = typename DEVICE::index_t;
        {

            float32_t *weights_row = layer.weights._data;
            float32_t *input_row = input._data;
            float32_t *output_row = output._data;

            float32_t *weights_element, *input_element, *output_element;

            float32_t sum;
            uint32_t weights_row_i, i = 0U, batch_i = BATCH_SIZE, input_i;

            {
                do{
                    output_element = output_row;

                    weights_row_i = LAYER_SPEC::OUTPUT_DIM;


                    do
                    {
                        sum = 0.0f;
                        input_element = input_row;
                        weights_element = weights_row;

                        input_i = ((uint32_t)LAYER_SPEC::INPUT_DIM) >> 2U;
                        while (input_i > 0U)
                        {
                            sum += *weights_element++ * *input_element++;
                            sum += *weights_element++ * *input_element++;
                            sum += *weights_element++ * *input_element++;
                            sum += *weights_element++ * *input_element++;

                            input_i--;
                        }

                        input_i = ((uint32_t)LAYER_SPEC::INPUT_DIM) % 0x4U;

                        while (input_i > 0U){
                            sum += *weights_element++ * *input_element++;
                            input_i--;
                        }

                        *output_element++ = sum;

                        weights_row_i--;

                        weights_row += decltype(layer.weights)::ROW_PITCH;

                    } while (weights_row_i > 0U);

                    output_row += OUTPUT_SPEC::ROW_PITCH;
                    input_row += INPUT_SPEC::ROW_PITCH;

                    batch_i--;

                } while (batch_i > 0U);
            }

        }

        arm_matrix_instance_f32 arm_weights = {
            .numRows = LAYER_SPEC::OUTPUT_DIM,
            .numCols = LAYER_SPEC::INPUT_DIM,
            .pData = layer.weights._data
        };

        arm_matrix_instance_f32 arm_input = {
            .numRows = LAYER_SPEC::INPUT_DIM,
            .numCols = BATCH_SIZE,
            .pData = input._data
        };
        arm_matrix_instance_f32 arm_output = {
            .numRows = LAYER_SPEC::OUTPUT_DIM,
            .numCols = BATCH_SIZE,
            .pData = output._data
        } ;
        // arm_mat_mult_f32(&arm_weights, &arm_input, &arm_output);
        // beware this only works for batch size = 1
        arm_add_f32(output._data, layer.biases._data, output._data, LAYER_SPEC::OUTPUT_DIM);


        for(TI i = 0; i < BATCH_SIZE; i++){
            for(TI j = 0; j < LAYER_SPEC::OUTPUT_DIM; j++){
                set(output, i, j, activation<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(get(output, i, j)));
            }
        }
    }
}

#endif

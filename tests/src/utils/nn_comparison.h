#ifndef BACKPROP_TOOLS_TESTS_UTILS_NN_COMPARISON_H
#define BACKPROP_TOOLS_TESTS_UTILS_NN_COMPARISON_H

//template <typename DEVICE, typename SPEC>
//typename SPEC::T abs_diff(const backprop_tools::nn::layers::dense::Layer<DEVICE, SPEC>& l1, const backprop_tools::nn::layers::dense::Layer<DEVICE, SPEC>& l2) {
//    typedef typename SPEC::T T;
//    T acc = 0;
//    acc += abs_diff_matrix<T, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(l1.weights, l2.weights);
//    acc += abs_diff_vector<T, SPEC::OUTPUT_DIM>(l1.biases, l2.biases);
//    return acc;
//}
//template <typename DEVICE, typename SPEC>
//typename SPEC::T abs_diff_grad(const backprop_tools::nn::layers::dense::LayerBackwardGradient<DEVICE, SPEC>& l1, const backprop_tools::nn::layers::dense::LayerBackwardGradient<DEVICE, SPEC>& l2) {
//    typedef typename SPEC::T T;
//    T acc = 0; //abs_diff((backprop_tools::nn::layers::dense::Layer<DEVICE, SPEC>&)l1, (backprop_tools::nn::layers::dense::Layer<DEVICE, SPEC>&)l2);
//    acc += abs_diff_matrix<T, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(l1.d_weights, l2.d_weights);
//    acc += abs_diff_vector<T, SPEC::OUTPUT_DIM>(l1.d_biases, l2.d_biases);
//    return acc;
//}
//template <typename DEVICE, typename SPEC, typename PARAMETERS>
//typename SPEC::T abs_diff_adam(const backprop_tools::nn::layers::dense::LayerBackwardAdam<DEVICE, SPEC, PARAMETERS>& l1, const backprop_tools::nn::layers::dense::LayerBackwardAdam<DEVICE, SPEC, PARAMETERS>& l2) {
//    typedef typename SPEC::T T;
//    T acc = 0; //abs_diff((backprop_tools::nn::layers::dense::LayerBackwardGradient<DEVICE, SPEC>&)l1, (backprop_tools::nn::layers::dense::LayerBackwardGradient<DEVICE, SPEC>&)l2);
//    acc += abs_diff_matrix<T, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(l1.d_weights_first_order_moment, l2.d_weights_first_order_moment);
//    acc += abs_diff_matrix<T, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(l1.d_weights_second_order_moment, l2.d_weights_second_order_moment);
//    acc += abs_diff_vector<T, SPEC::OUTPUT_DIM>(l1.d_biases_first_order_moment, l2.d_biases_first_order_moment);
//    acc += abs_diff_vector<T, SPEC::OUTPUT_DIM>(l1.d_biases_second_order_moment, l2.d_biases_second_order_moment);
//    return acc;
//}
#endif
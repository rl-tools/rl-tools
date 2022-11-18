
template<typename T, int INPUT_DIM, int LAYER_1_DIM, int LAYER_2_DIM, int OUTPUT_DIM>
void init(ThreeLayerNeuralNetworkInference<T, INPUT_DIM, LAYER_1_DIM, LAYER_2_DIM, OUTPUT_DIM>& policy){
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<T> dist(-0.1, 0.1);

    for(int i = 0; i < LAYER_1_DIM; i++) {
        for(int j = 0; j < INPUT_DIM; j++) {
            policy.layer_0.weights[i][j] = dist(rng);
        }
        policy.layer_0.biases[i] = 0;
    }
    for(int i = 0; i < LAYER_2_DIM; i++) {
        for(int j = 0; j < LAYER_1_DIM; j++) {
            policy.layer_1.weights[i][j] = dist(rng);
        }
        policy.layer_1.biases[i] = 0;
    }
    for(int i = 0; i < OUTPUT_DIM; i++) {
        for(int j = 0; j < LAYER_2_DIM; j++) {
            policy.layer_2.weights[i][j] = dist(rng);
        }
        policy.layer_2.biases[i] = 0;
    }
}

template<typename T, int INPUT_DIM, int OUTPUT_DIM>
struct ConstantNeuralNetwork{
};

template<typename T, int INPUT_DIM, int OUTPUT_DIM>
FUNCTION_PLACEMENT void evaluate(const ConstantNeuralNetwork<T, INPUT_DIM, OUTPUT_DIM>& network, const T input[INPUT_DIM], T output[OUTPUT_DIM]) {
//    for(int i = 0; i < OUTPUT_DIM; i++) {
//        output[i] = 0;
//    }
    output[0] = 1000;
    output[1] = 500;
    output[2] = 750;
    output[3] = 250;
//    output[0] = 1000;
//    output[1] = 1000;
//    output[2] = 1000;
//    output[3] = 1000;
}

import * as hdf5 from "https://esm.sh/jsfive";
import * as math from 'https://esm.sh/mathjs'

class Matrix{
    constructor(dataset){
        this.rows = dataset.shape[0]
        this.cols = dataset.shape[1]
        const data_flat = math.matrix(dataset.value)
        this.data = math.reshape(data_flat, [this.rows, this.cols])
    }
}

class Tensor{
    constructor(dataset){
        this.shape = dataset.shape
        const data_flat = math.matrix(dataset.value)
        this.data = math.reshape(data_flat, this.shape)
    }
}

class DenseLayer{
    constructor(group){
        this.weights = group.get("weights").attrs.type === "matrix" ? new Matrix(group.get("weights").get("parameters")) : new Tensor(group.get("weights").get("parameters"))
        this.biases = group.get("biases").attrs.type === "matrix" ? new Matrix(group.get("biases").get("parameters")) : new Tensor(group.get("biases").get("parameters"))
        this.activation_function_name = group.attrs.activation_function
    }
    activation_function(input){
        if(this.activation_function_name === "IDENTITY"){
            return input
        }
        else if(this.activation_function_name === "RELU"){
            return math.map(input, x => x > 0 ? x : 0)
        }
        else if(this.activation_function_name === "SIGMOID"){
            return math.map(input, x => 1 / (1 + math.exp(-x)))
        }
        else if(this.activation_function_name === "TANH"){
            return math.map(input, x => math.tanh(x))
        }
        else{
            console.error("Unknown activation function: ", this.activation_function_name)
            return null
        }
    }
    evaluate(input){
        const leading_dimension = input.size().slice(0, -1).reduce((a, b) => a * b, 1)
        const input_reshaped = math.reshape(input, [leading_dimension, input.size()[input.size().length - 1]])
        let output = math.multiply(this.weights.data, math.transpose(input_reshaped))
        output = math.add(math.transpose(output), this.biases.data)
        output = this.activation_function(output)
        const output_shape = input.size().slice(0, -1).concat(this.biases.shape[1])
        output = math.reshape(output, output_shape)
        return output
    }
}
class SampleAndSquashLayer{
    constructor(group){
    }
    evaluate(input){
        const mean = math.subset(input, math.index(
            ...input.size().map((x, i) => {
                if(i === input.size().length - 1){
                    return math.range(0, x/2)
                }
                else{
                    return math.range(0, x)
                }
            })
        ));
        return math.map(mean, Math.tanh)
    }
}
class MLP{
    constructor(group){
        this.input_layer = new DenseLayer(group.get("input_layer"))
        this.hidden_layers = []
        for(let i = 0; i < group.attrs.num_layers - 2; i++){
            this.hidden_layers.push(new DenseLayer(group.get(`hidden_layer_${i}`)))
        }
        this.output_layer = new DenseLayer(group.get("output_layer"))
    }
    evaluate(input){
        let current = this.input_layer.evaluate(input)
        for(let i = 0; i < this.hidden_layers.length; i++){
            const layer = this.hidden_layers[i]
            current = layer.evaluate(current)
        }
        current = this.output_layer.evaluate(current)
        return current
    }
}

class Sequential{
    constructor(group){
        this.layers = []
        for(let i = 0; i < group.get("layers").keys.length; i++){
            this.layers.push(layer_dispatch(group.get("layers").get(`${i}`)))
        }
    }
    evaluate(input){
        let current = input
        for(let i = 0; i < this.layers.length; i++){
            const layer = this.layers[i]
            if(layer){
                current = layer.evaluate(current)
            }
        }
        return current
    }
}


function layer_dispatch(group){
    if(group.attrs.type === "dense") {
        return new DenseLayer(group)
    }
    else if(group.attrs.type === "mlp") {
        return new MLP(group)
    }
    else if(group.attrs.type === "sequential") {
        return new Sequential(group)
    }
    else if(group.attrs.type === "sample_and_squash") {
        return new SampleAndSquashLayer(group)
    }
    else{
        console.error("Unknown layer type: ", group.attrs.type)
        return null
    }
}

function load_from_array_buffer(buffer){
    var f = new hdf5.File(buffer, "");
    const model = layer_dispatch(f.get("actor"))
    const input = new Tensor(f.get("example").get("input"))
    const target_output = new Tensor(f.get("example").get("output"))

    const output = model.evaluate(input.data)

    const diff = math.subtract(output, target_output.data)
    const diff_reduce = math.flatten(diff).valueOf().reduce((a, c) => a + Math.abs(c)) / diff.size().reduce((a, c) => a * c, 1)
    console.log("Example diff per element: ", diff_reduce)
    console.assert(diff_reduce < 1e-6, "Output is not close enough to target output")
    return model
}

export function load(input) {
    if(typeof input === "string"){
        return fetch(input)
            .then(function(response) {
                return response.arrayBuffer()
            })
            .then(load_from_array_buffer);
    }
    else if(input instanceof ArrayBuffer){
        return load_from_array_buffer(input)
    }
    else{
        console.error("Input is not a string or ArrayBuffer")
        return null
    }
}


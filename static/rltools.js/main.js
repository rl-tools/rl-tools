import * as hdf5 from "jsfive";
import * as math from 'mathjs'

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

function apply_activation(name, input){
    if(name === "IDENTITY"){
        return input
    }
    else if(name === "RELU"){
        return math.map(input, x => x > 0 ? x : 0)
    }
    else if(name === "GELU"){
        return math.map(input, x => {
            return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)))
        })
    }
    else if(name === "SIGMOID"){
        return math.map(input, x => 1 / (1 + Math.exp(-x)))
    }
    else if(name === "TANH"){
        return math.map(input, x => Math.tanh(x))
    }
    else if(name === "FAST_TANH"){
        return math.map(input, x => {
            x = Math.max(-3.0, Math.min(3.0, x))
            const x_squared = x * x
            return x * (27 + x_squared) / (27 + 9 * x_squared)
        })
    }
    else{
        console.error("Unknown activation function: ", name)
        return null
    }
}

class StandardizeLayer{
    constructor(group){
        this.mean = group.get("mean").attrs.type === "matrix" ? new Matrix(group.get("mean").get("parameters")) : new Tensor(group.get("mean").get("parameters"))
        this.dim = this.mean.shape.length == 2 ? this.mean.shape[1] : this.mean.shape[0]
        this.input_shape = [null, null, this.dim]
        this.output_shape = [null, null, this.dim]
        this.precision = group.get("precision").attrs.type === "matrix" ? new Matrix(group.get("precision").get("parameters")) : new Tensor(group.get("precision").get("parameters"))
    }
    description(){
        return `Standardize(${this.output_shape[2]})`
    }
    evaluate(input){
        const leading_dimension = input.size().slice(0, -1).reduce((a, b) => a * b, 1)
        const input_reshaped = math.reshape(input, [leading_dimension, input.size()[input.size().length - 1]])
        let [output, state] = this.evaluate_step(input_reshaped)
        const output_shape = input.size().slice(0, -1).concat(this.dim)
        output = math.reshape(output, output_shape)
        return output
    }
    evaluate_step(input, state){
        let output = math.dotMultiply(math.subtract(input, this.mean.data), this.precision.data)
        return [output, null]
    }
}

class DenseLayer{
    constructor(group){
        this.weights = group.get("weights").attrs.type === "matrix" ? new Matrix(group.get("weights").get("parameters")) : new Tensor(group.get("weights").get("parameters"))
        this.input_shape = [null, null, this.weights.shape[1]]
        this.output_shape = [null, null, this.weights.shape[0]]
        this.biases = group.get("biases").attrs.type === "matrix" ? new Matrix(group.get("biases").get("parameters")) : new Tensor(group.get("biases").get("parameters"))
        this.dim = this.biases.shape.length == 2 ? this.biases.shape[1] : this.biases.shape[0]
        this.activation_function_name = group.attrs.activation_function
    }
    description(){
        return `Dense(${this.output_shape[2]})`
    }
    evaluate(input){
        const leading_dimension = input.size().slice(0, -1).reduce((a, b) => a * b, 1)
        const input_reshaped = math.reshape(input, [leading_dimension, input.size()[input.size().length - 1]])
        let [output, state] = this.evaluate_step(input_reshaped)
        const output_shape = input.size().slice(0, -1).concat(this.dim)
        output = math.reshape(output, output_shape)
        return output
    }
    evaluate_step(input, state){
        let output = math.multiply(this.weights.data, math.transpose(input))
        output = math.add(math.transpose(output), this.biases.data)
        output = apply_activation(this.activation_function_name, output)
        return [output, null]
    }
}

class Conv2dLayer{
    constructor(group){
        this.weights = new Tensor(group.get("weights").get("parameters"))
        this.biases = new Tensor(group.get("biases").get("parameters"))
        this.output_channels = parseInt(group.attrs.output_channels)
        this.input_channels = parseInt(group.attrs.input_channels)
        this.kernel_height = parseInt(group.attrs.kernel_height)
        this.kernel_width = parseInt(group.attrs.kernel_width)
        this.stride_h = parseInt(group.attrs.stride_h)
        this.stride_w = parseInt(group.attrs.stride_w)
        this.padding_h = parseInt(group.attrs.padding_h)
        this.padding_w = parseInt(group.attrs.padding_w)
        this.activation_function_name = group.attrs.activation_function
        this.normalization = group.attrs.normalization || "NONE"
        this.input_shape = [null, null, null, this.input_channels]
        this.output_shape = [null, null, null, this.output_channels]
        if(this.normalization !== "NONE"){
            this.gamma = new Tensor(group.get("gamma").get("parameters"))
            this.beta = new Tensor(group.get("beta").get("parameters"))
            if(this.normalization === "BATCH_NORM"){
                this.running_mean = new Tensor(group.get("running_mean"))
                this.running_var = new Tensor(group.get("running_var"))
            }
        }
    }
    description(){
        return `Conv2d(${this.output_channels}, ${this.kernel_height}x${this.kernel_width})`
    }
    evaluate(input){
        const size = input.size()
        const rank = size.length
        const input_height = size[rank - 3]
        const input_width = size[rank - 2]
        const output_height = Math.floor((input_height + 2 * this.padding_h - this.kernel_height) / this.stride_h) + 1
        const output_width = Math.floor((input_width + 2 * this.padding_w - this.kernel_width) / this.stride_w) + 1
        const leading_dims = size.slice(0, rank - 3)
        const batch_size = leading_dims.reduce((a, b) => a * b, 1)
        const input_flat = math.reshape(input, [batch_size, input_height, input_width, this.input_channels]).valueOf()
        const weights = this.weights.data.valueOf()
        const biases = this.biases.data.valueOf()
        const output_flat = new Array(batch_size)
        for(let bi = 0; bi < batch_size; bi++){
            output_flat[bi] = new Array(output_height)
            for(let oh = 0; oh < output_height; oh++){
                output_flat[bi][oh] = new Array(output_width)
                for(let ow = 0; ow < output_width; ow++){
                    output_flat[bi][oh][ow] = new Array(this.output_channels)
                    for(let oc = 0; oc < this.output_channels; oc++){
                        let acc = biases[oc]
                        for(let kh = 0; kh < this.kernel_height; kh++){
                            const ih = oh * this.stride_h + kh - this.padding_h
                            if(ih < 0 || ih >= input_height) continue
                            for(let kw = 0; kw < this.kernel_width; kw++){
                                const iw = ow * this.stride_w + kw - this.padding_w
                                if(iw < 0 || iw >= input_width) continue
                                for(let ic = 0; ic < this.input_channels; ic++){
                                    acc += weights[oc][kh][kw][ic] * input_flat[bi][ih][iw][ic]
                                }
                            }
                        }
                        output_flat[bi][oh][ow][oc] = acc
                    }
                }
            }
        }
        let result = math.matrix(output_flat)
        if(this.normalization === "BATCH_NORM"){
            const running_mean = this.running_mean.data.valueOf()
            const running_var = this.running_var.data.valueOf()
            const gamma = this.gamma.data.valueOf()
            const beta = this.beta.data.valueOf()
            const epsilon = 1e-5
            const normalized = new Array(batch_size)
            for(let bi = 0; bi < batch_size; bi++){
                normalized[bi] = new Array(output_height)
                for(let oh = 0; oh < output_height; oh++){
                    normalized[bi][oh] = new Array(output_width)
                    for(let ow = 0; ow < output_width; ow++){
                        normalized[bi][oh][ow] = new Array(this.output_channels)
                        for(let oc = 0; oc < this.output_channels; oc++){
                            const x = output_flat[bi][oh][ow][oc]
                            normalized[bi][oh][ow][oc] = gamma[oc] * (x - running_mean[oc]) / Math.sqrt(running_var[oc] + epsilon) + beta[oc]
                        }
                    }
                }
            }
            result = math.matrix(normalized)
        }
        else if(this.normalization === "LAYER_NORM"){
            const gamma = this.gamma.data.valueOf()
            const beta = this.beta.data.valueOf()
            const epsilon = 1e-5
            const normalized = new Array(batch_size)
            for(let bi = 0; bi < batch_size; bi++){
                let sum = 0
                let count = output_height * output_width * this.output_channels
                for(let oh = 0; oh < output_height; oh++){
                    for(let ow = 0; ow < output_width; ow++){
                        for(let oc = 0; oc < this.output_channels; oc++){
                            sum += output_flat[bi][oh][ow][oc]
                        }
                    }
                }
                const mean = sum / count
                let var_sum = 0
                for(let oh = 0; oh < output_height; oh++){
                    for(let ow = 0; ow < output_width; ow++){
                        for(let oc = 0; oc < this.output_channels; oc++){
                            const d = output_flat[bi][oh][ow][oc] - mean
                            var_sum += d * d
                        }
                    }
                }
                const inv_std = 1 / Math.sqrt(var_sum / count + epsilon)
                normalized[bi] = new Array(output_height)
                for(let oh = 0; oh < output_height; oh++){
                    normalized[bi][oh] = new Array(output_width)
                    for(let ow = 0; ow < output_width; ow++){
                        normalized[bi][oh][ow] = new Array(this.output_channels)
                        for(let oc = 0; oc < this.output_channels; oc++){
                            normalized[bi][oh][ow][oc] = gamma[oc] * (output_flat[bi][oh][ow][oc] - mean) * inv_std + beta[oc]
                        }
                    }
                }
            }
            result = math.matrix(normalized)
        }
        result = apply_activation(this.activation_function_name, result)
        const output_shape = leading_dims.concat([output_height, output_width, this.output_channels])
        return math.reshape(result, output_shape)
    }
    evaluate_step(input, state){
        return [this.evaluate(input), null]
    }
}

class FlattenLayer{
    constructor(group){
        this.input_shape = [null, null, null, null]
        this.output_shape = [null, null]
    }
    description(){
        return `Flatten`
    }
    evaluate(input){
        const size = input.size()
        const rank = size.length
        if(rank < 3) return input
        const leading_dims = size.slice(0, rank - 3)
        const flat_dim = size[rank - 3] * size[rank - 2] * size[rank - 1]
        return math.reshape(input, leading_dims.concat([flat_dim]))
    }
    evaluate_step(input, state){
        return [this.evaluate(input), null]
    }
}

class AvgPool2dLayer{
    constructor(group){
        this.input_shape = [null, null, null, null]
        this.output_shape = [null, null]
    }
    description(){
        return `AvgPool2d`
    }
    evaluate(input){
        const size = input.size()
        const rank = size.length
        const height = size[rank - 3]
        const width = size[rank - 2]
        const channels = size[rank - 1]
        const leading_dims = size.slice(0, rank - 3)
        const batch_size = leading_dims.reduce((a, b) => a * b, 1)
        const input_flat = math.reshape(input, [batch_size, height, width, channels]).valueOf()
        const scale = 1.0 / (height * width)
        const output_flat = new Array(batch_size)
        for(let bi = 0; bi < batch_size; bi++){
            output_flat[bi] = new Array(channels).fill(0)
            for(let h = 0; h < height; h++){
                for(let w = 0; w < width; w++){
                    for(let c = 0; c < channels; c++){
                        output_flat[bi][c] += input_flat[bi][h][w][c] * scale
                    }
                }
            }
        }
        return math.reshape(math.matrix(output_flat), leading_dims.concat([channels]))
    }
    evaluate_step(input, state){
        return [this.evaluate(input), null]
    }
}

class MaxPool2dLayer{
    constructor(group){
        this.kernel_height = parseInt(group.attrs.kernel_height)
        this.kernel_width = parseInt(group.attrs.kernel_width)
        this.stride_h = parseInt(group.attrs.stride_h)
        this.stride_w = parseInt(group.attrs.stride_w)
        this.padding_h = parseInt(group.attrs.padding_h)
        this.padding_w = parseInt(group.attrs.padding_w)
        this.input_shape = [null, null, null, null]
        this.output_shape = [null, null, null, null]
    }
    description(){
        return `MaxPool2d(${this.kernel_height}x${this.kernel_width})`
    }
    evaluate(input){
        const size = input.size()
        const rank = size.length
        const input_height = size[rank - 3]
        const input_width = size[rank - 2]
        const channels = size[rank - 1]
        const output_height = Math.floor((input_height + 2 * this.padding_h - this.kernel_height) / this.stride_h) + 1
        const output_width = Math.floor((input_width + 2 * this.padding_w - this.kernel_width) / this.stride_w) + 1
        const leading_dims = size.slice(0, rank - 3)
        const batch_size = leading_dims.reduce((a, b) => a * b, 1)
        const input_flat = math.reshape(input, [batch_size, input_height, input_width, channels]).valueOf()
        const output_flat = new Array(batch_size)
        for(let bi = 0; bi < batch_size; bi++){
            output_flat[bi] = new Array(output_height)
            for(let oh = 0; oh < output_height; oh++){
                output_flat[bi][oh] = new Array(output_width)
                for(let ow = 0; ow < output_width; ow++){
                    output_flat[bi][oh][ow] = new Array(channels)
                    for(let c = 0; c < channels; c++){
                        let max_val = -Infinity
                        for(let kh = 0; kh < this.kernel_height; kh++){
                            const ih = oh * this.stride_h + kh - this.padding_h
                            if(ih < 0 || ih >= input_height) continue
                            for(let kw = 0; kw < this.kernel_width; kw++){
                                const iw = ow * this.stride_w + kw - this.padding_w
                                if(iw < 0 || iw >= input_width) continue
                                const val = input_flat[bi][ih][iw][c]
                                if(val > max_val) max_val = val
                            }
                        }
                        output_flat[bi][oh][ow][c] = max_val
                    }
                }
            }
        }
        return math.reshape(math.matrix(output_flat), leading_dims.concat([output_height, output_width, channels]))
    }
    evaluate_step(input, state){
        return [this.evaluate(input), null]
    }
}

class GRULayer{
    constructor(group){
        this.weights_hidden = new Tensor(group.get("weights_hidden").get("parameters"))
        this.weights_input = new Tensor(group.get("weights_input").get("parameters"))
        this.hidden_dim = Math.floor(this.weights_input.shape[0] / 3)
        this.input_shape = [null, null, this.weights_input.shape[1]]
        this.output_shape = [null, null, this.hidden_dim]
        this.biases_hidden = new Tensor(group.get("biases_hidden").get("parameters"))
        this.biases_input = new Tensor(group.get("biases_input").get("parameters"))
        this.initial_hidden_state = new Tensor(group.get("initial_hidden_state").get("parameters"))
    }
    description(){
        return `GRU(${this.hidden_dim})`
    }
    reset(){
        return null
    }
    evaluate(input){
        if(input.size().length === 2){
            const [output, state] = this.evaluate_step(input)
            return output
        }
        else{
            const [SEQUENCE_LENGTH, BATCH_SIZE, INPUT_SIZE] = input.size()
            let state = null
            const outputs = []
            for(let i = 0; i < SEQUENCE_LENGTH; i++){
                const step_input = math.matrix((math.subset(input, math.index(i, math.range(0, BATCH_SIZE), math.range(0, INPUT_SIZE)))).toArray()[0])
                const [output, new_state] = this.evaluate_step(step_input, state)
                outputs.push(output)
                state = new_state
            }
            return math.matrix(outputs)
        }
    }
    evaluate_step(input, state){
        const [BATCH_SIZE, INPUT_SIZE] = input.size()
        console.assert(INPUT_SIZE === this.weights_input.shape[1], "Input size does not match weights size")
        if(state === null){
            state = math.matrix((new Array(BATCH_SIZE)).fill(0).map(() => this.initial_hidden_state.data))
        }
        console.assert(state.size()[0] === BATCH_SIZE, "State size does not match input size")
        const Wh = math.transpose(math.multiply(this.weights_hidden.data, math.transpose(state)))
        const Wi = math.transpose(math.multiply(this.weights_input.data, math.transpose(input)))
        const Wh_rz = math.subset(Wh, math.index(math.range(0, BATCH_SIZE), math.range(0, this.hidden_dim * 2)))
        const Wh_n = math.subset(Wh, math.index(math.range(0, BATCH_SIZE), math.range(this.hidden_dim * 2, this.hidden_dim * 3)))
        const Wi_rz = math.subset(Wi, math.index(math.range(0, BATCH_SIZE), math.range(0, this.hidden_dim * 2)))
        const Wi_n = math.subset(Wi, math.index(math.range(0, BATCH_SIZE), math.range(this.hidden_dim * 2, this.hidden_dim * 3)))
        const bh_rz = math.subset(this.biases_hidden.data, math.index(math.range(0, this.hidden_dim * 2)))
        const bh_n = math.subset(this.biases_hidden.data, math.index(math.range(this.hidden_dim * 2, this.hidden_dim * 3)))
        const bi_rz = math.subset(this.biases_input.data, math.index(math.range(0, this.hidden_dim * 2)))
        const bi_n = math.subset(this.biases_input.data, math.index(math.range(this.hidden_dim * 2, this.hidden_dim * 3)))
        const rz_pre = math.add(math.add(math.add(Wh_rz, Wi_rz), bh_rz), bi_rz)
        const sigmoid = (x) => 1 / (1 + Math.exp(-x))
        const rz = math.map(rz_pre, sigmoid)
        const r = math.subset(rz, math.index(math.range(0, BATCH_SIZE), math.range(0, this.hidden_dim)))
        const z = math.subset(rz, math.index(math.range(0, BATCH_SIZE), math.range(this.hidden_dim, this.hidden_dim * 2)))
        const n_pre_pre = math.add(Wh_n, bh_n)
        const n_pre = math.add(math.add(math.dotMultiply(r, n_pre_pre), Wi_n), bi_n)
        const n = math.map(n_pre, Math.tanh)
        const new_state = math.add(math.dotMultiply(z, state), math.dotMultiply(math.subtract(1, z), n))
        return [new_state, new_state]
    }
}
class SampleAndSquashLayer{
    constructor(group){
        this.input_shape = [null, null, null]
        this.output_shape = [null, null, null]
    }
    description(){
        return `SampleAndSquash`
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
    evaluate_step(input, state){
        return [this.evaluate(input), null]
    }
}
class MLP{
    constructor(group){
        this.input_layer = new DenseLayer(group.get("input_layer"))
        this.hidden_layers = []
        for(let i = 0; i < group.attrs.num_layers - 2; i++){
            if(group.keys.includes(`hidden_layer_${i}`)){
                this.hidden_layers.push(new DenseLayer(group.get(`hidden_layer_${i}`)))
            }
            else{
                const hidden_layers_group = group.get("hidden_layers")
                this.hidden_layers.push(new DenseLayer(hidden_layers_group.get(`${i}`)))
            }
        }
        this.output_layer = new DenseLayer(group.get("output_layer"))
        this.input_shape = this.input_layer.input_shape
        this.output_shape = this.output_layer.output_shape
    }
    description(){
        return `MLP(${this.input_layer.description()}, ${this.hidden_layers.map(layer => layer.description()).join(", ")}, ${this.output_layer.description()})`
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
    evaluate_step(input, state){
        return [this.evaluate(input), null]
    }
}

class Sequential{
    constructor(group){
        this.layers = []
        for(let i = 0; i < group.get("layers").keys.length; i++){
            this.layers.push(layer_dispatch(group.get("layers").get(`${i}`)))
        }
        this.input_shape = this.layers[0].input_shape
        this.output_shape = this.layers.slice().reverse().find(layer => layer.output_shape.reduce((a, c) => (a || c !== null), null)).output_shape
    }
    description(){
        return `Sequential(${this.layers.map(layer => layer.description()).join(", ")})`
    }
    reset(){
        return this.layers.map(layer => {
            return layer.reset ? layer.reset() : null
        })
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
    evaluate_step(input, state){
        if(!state){
            state = this.reset()
        }
        let current = input
        const new_state = []
        for(let i = 0; i < this.layers.length; i++){
            const layer = this.layers[i]
            const layer_state = state[i]
            if(layer){
                const [new_current, new_layer_state] = layer.evaluate_step(current, layer_state)
                current = new_current
                new_state.push(new_layer_state)
            }
            else{
                new_state.push(null)
            }
        }
        return [current, new_state]
    }
}

class Parallel{
    constructor(group){
        this.pipeline_a = layer_dispatch(group.get("pipeline_a"))
        this.pipeline_b = layer_dispatch(group.get("pipeline_b"))
        this.head = null
        if(group.keys.includes("head")){
            this.head = layer_dispatch(group.get("head"))
        }
        this.input_shape = [null]
        this.output_shape = this.head ? this.head.output_shape : [null]
    }
    description(){
        const head_str = this.head ? `, ${this.head.description()}` : ""
        return `Parallel(${this.pipeline_a.description()}, ${this.pipeline_b.description()}${head_str})`
    }
    evaluate(input_a, input_b){
        const out_a = this.pipeline_a.evaluate(input_a)
        const out_b = this.pipeline_b.evaluate(input_b)
        const size_a = out_a.size()
        const size_b = out_b.size()
        const last_dim_a = size_a[size_a.length - 1]
        const last_dim_b = size_b[size_b.length - 1]
        const leading_dims = size_a.slice(0, -1)
        const leading_count = leading_dims.reduce((a, b) => a * b, 1)
        const flat_a = math.reshape(out_a, [leading_count, last_dim_a]).valueOf()
        const flat_b = math.reshape(out_b, [leading_count, last_dim_b]).valueOf()
        const concatenated = new Array(leading_count)
        for(let i = 0; i < leading_count; i++){
            concatenated[i] = flat_a[i].concat(flat_b[i])
        }
        let result = math.reshape(math.matrix(concatenated), leading_dims.concat([last_dim_a + last_dim_b]))
        if(this.head){
            result = this.head.evaluate(result)
        }
        return result
    }
    evaluate_step(input_a, input_b, state){
        return [this.evaluate(input_a, input_b), null]
    }
}


function layer_dispatch(group){
    let model = null
    if(group.attrs.type === "dense") {
        model = new DenseLayer(group)
    }
    else if(group.attrs.type === "conv2d") {
        model = new Conv2dLayer(group)
    }
    else if(group.attrs.type === "flatten") {
        model = new FlattenLayer(group)
    }
    else if(group.attrs.type === "avg_pool2d") {
        model = new AvgPool2dLayer(group)
    }
    else if(group.attrs.type === "max_pool2d") {
        model = new MaxPool2dLayer(group)
    }
    else if(group.attrs.type === "gru") {
        model = new GRULayer(group)
    }
    else if(group.attrs.type === "mlp") {
        model = new MLP(group)
    }
    else if(group.attrs.type === "sequential") {
        model = new Sequential(group)
    }
    else if(group.attrs.type === "parallel") {
        model = new Parallel(group)
    }
    else if(group.attrs.type === "sample_and_squash") {
        model = new SampleAndSquashLayer(group)
    }
    else if(group.attrs.type === "standardize") {
        model = new StandardizeLayer(group)
    }
    else{
        console.error("Unknown layer type: ", group.attrs.type)
        model = null
    }
    model.checkpoint_name = null
    if("checkpoint_name" in group.attrs){
        model.checkpoint_name = group.attrs.checkpoint_name
    }
    else{
    }
    model.meta = null
    if("meta" in group.attrs){
        model.meta = JSON.parse(group.attrs.meta)
    }
    return model
}

function load_from_array_buffer(buffer){
    var f = new hdf5.File(buffer, "");
    const model = layer_dispatch(f.get("actor"))
    const inputs_group = f.get("example/inputs")
    const outputs_group = f.get("example/outputs")

    const input_keys = [...inputs_group.keys].sort((a, b) => parseInt(a) - parseInt(b))
    const input_datas = input_keys.map(k => new Tensor(inputs_group.get(k)).data)
    const output = model.evaluate(...input_datas)

    const target_output = new Tensor(outputs_group.get("0"))
    const diff = math.subtract(output, target_output.data)
    const diff_reduce = math.flatten(diff).valueOf().reduce((a, c) => a + Math.abs(c)) / diff.size().reduce((a, c) => a * c, 1)
    console.log("Example diff per element: ", diff_reduce)
    console.assert(diff_reduce < 1e-5, "Output is not close enough to target output")
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

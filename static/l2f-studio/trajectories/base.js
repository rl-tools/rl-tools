export class Trajectory{
    constructor(parameters){
        this.parameters = { "length": {"range": [1, 60], "default": 30, "step": 0.5}, ...parameters }
        this.parameter_values = {}
        for(const [name, param] of Object.entries(this.parameters)){
            this.parameter_values[name] = param.default
        }
        this.sample_rate = 100 // Hz = control frequency
        this.onUpdate = null
        this.parameters_updated()
    }
    set_parameter(name, value){
        this.parameter_values[name] = value
    }
    reset(){
        this.parameter_values = {}
        for(const [name, param] of Object.entries(this.parameters)){
            this.parameter_values[name] = param.default
        }
    }
    parameters_updated(){
        const length = this.parameter_values.length
        const num_steps = Math.round(length * this.sample_rate)
        this.trajectory = new Array(num_steps).fill(0).map((_, i) => {
            return this.evaluate(i / this.sample_rate)
        })
        this.onUpdate?.()
    }
}
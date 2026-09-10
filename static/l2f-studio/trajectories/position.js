import { Trajectory } from "./base.js"
export class Position extends Trajectory{
    constructor(){
        super({
            "x": {"range": [-10, 10], "default": 0},
            "y": {"range": [-10, 10], "default": 0},
            "z": {"range": [-10, 10], "default": 0},
        })
    }
    evaluate(t){
        return [this.parameter_values.x, this.parameter_values.y, this.parameter_values.z, 0, 0, 0]
    }
}

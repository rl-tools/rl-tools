
import { Trajectory } from "./base.js"
export class PingPong extends Trajectory{
    constructor(){
        super({
            "period": {"range": [1, 15], "default": 5},
            "scale": {"range": [0, 5], "default": 0.1},
        })
    }
    evaluate(t){
        const scale = this.parameter_values.scale
        const duration = this.parameter_values.period
        const progress = t % duration
        const ping = progress < duration / 2
        return [ping ? scale : - scale, 0, 0, 0, 0, 0]
    }
}
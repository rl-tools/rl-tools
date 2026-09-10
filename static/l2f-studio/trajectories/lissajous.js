import { Trajectory } from "./base.js"
export class Lissajous extends Trajectory{
    constructor(){
        super({
            "A": {"range": [0, 2], "default": 0.5},
            "B": {"range": [0, 2], "default": 1.0},
            "C": {"range": [0, 2], "default": 0.0},
            "a": {"range": [0, 5], "default": 2.0},
            "b": {"range": [0, 5], "default": 1.0},
            "c": {"range": [0, 5], "default": 1.0},
            "interval": {"range": [1, 30], "default": 10.0, "step": 0.5},
            "ramp": {"range": [0, 10], "default": 3.0},
        })
    }
    evaluate(t){
        const { A, B, C, a, b, c, interval, ramp } = this.parameter_values
        const time_velocity = ramp > 0 ? Math.min(t, ramp) / ramp : 1.0
        const ramp_time = time_velocity * Math.min(t, ramp) / 2.0
        const progress = (ramp_time + Math.max(0, t - ramp)) * 2 * Math.PI / interval 
        const d_progress = 2 * Math.PI * time_velocity / interval
        return [
            A * Math.sin(a * progress),
            B * Math.sin(b * progress),
            C * Math.sin(c * progress),
            A * Math.cos(a * progress) * a * d_progress,
            B * Math.cos(b * progress) * b * d_progress,
            C * Math.cos(c * progress) * c * d_progress
        ]
    }
}
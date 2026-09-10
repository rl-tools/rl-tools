import { Trajectory } from "./base.js"
import { Rand } from "rand-seed";


export class SecondOrderLangevin extends Trajectory {
    constructor() {
        super({
            // Damping
            gamma:  { range: [0, 5],  default: 1.0 },
            // Natural frequency
            omega:  { range: [0, 10], default: 2.0 },
            // Noise strength
            sigma:  { range: [0, 5],  default: 0.5 },
            // Periodicity
            T:      { range: [0, 20], default: 10.0 },
            // Time step
            dt:     { range: [0.001, 0.1], default: 0.01 },
            // EMA alpha
            alpha:  { range: [0, 1], default: 0.01 },
            // seed
            seed:  { range: [0, 10000], default: 1337 }
        })
    }

    parameters_updated() {
        // Precompute the Langevin dynamics first, then generate trajectory
        this.precompute()
        super.parameters_updated()
    }

    precompute() {
        const gamma = this.parameter_values.gamma
        const omega = this.parameter_values.omega
        const sigma = this.parameter_values.sigma
        const T     = this.parameter_values.T
        const dt    = this.parameter_values.dt
        const alpha = this.parameter_values.alpha

        this.N = Math.floor(T / dt) + 1

        this.P_raw = Array(this.N).fill().map(() => new Array(3))
        this.V_raw = Array(this.N).fill().map(() => new Array(3))
        this.P = Array(this.N).fill().map(() => new Array(3))
        this.V = Array(this.N).fill().map(() => new Array(3))

        for(let dim_i=0; dim_i < 3; dim_i++){
            this.P_raw[0][dim_i] = 0
            this.V_raw[0][dim_i] = 0
            this.P[0][dim_i] = 0
            this.V[0][dim_i] = 0
        }
        const rng = new Rand(this.parameter_values.seed.toString()) // rand-seed takes a string as seed for whatever cursed reason

        for (let step_i = 1; step_i < this.N; ++step_i) {
            for(let dim_i=0; dim_i<3; dim_i++) {
                const noise = randomGaussian(rng) * Math.sqrt(dt);
                const v_raw = this.V_raw[step_i-1][dim_i] + (-gamma*this.V_raw[step_i-1][dim_i] - omega*omega*this.P_raw[step_i-1][dim_i]) * dt + sigma * noise;
                this.V_raw[step_i][dim_i] = v_raw;
                this.P_raw[step_i][dim_i] = this.P_raw[step_i-1][dim_i] + v_raw * dt;
                const v_smooth = alpha * v_raw + (1-alpha) * this.V[step_i-1][dim_i];
                this.V[step_i][dim_i] = v_smooth;
                this.P[step_i][dim_i] = this.P[step_i-1][dim_i] + v_smooth * dt;
            }
        }
    }

    evaluate(t) {
        const dt = this.parameter_values.dt
        const T  = this.parameter_values.T
        const tr = reflectTime(t, T)
        let i = Math.floor(tr / dt)
        if (i < 0) i = 0
        if (i >= this.N) i = this.N - 1
        return [
            this.P[i][0],
            this.P[i][1],
            this.P[i][2],
            this.V[i][0],
            this.V[i][1],
            this.V[i][2],
        ]
    }
}

function randomGaussian(rng) {
    let u1 = rng.next()
    let u2 = rng.next()
    let r  = Math.sqrt(-2.0 * Math.log(u1))
    let theta = 2.0 * Math.PI * u2
    return r * Math.cos(theta)
}
function reflectTime(t, T) {
    const c = 2 * T
    let m = ((t % c) + c) % c
    return m <= T ? m : c - m
}
import { vec3, mat3, vec4, mat4, quat } from 'gl-matrix'; // vendored through the importmap in index.html
export class GamepadController{
    constructor(gamepad){
        this.actions = null
        this.actions_time = null
        this.k_omega = 1.0;
        gamepad.addListener((output) => {
            this.actions = output
            this.actions_time = performance.now()
        })
    }
    evaluate_step(state_bindings){
        return state_bindings.map((state_binding, i) => {
            const state = JSON.parse(state_binding.get_state());
            const params = JSON.parse(state_binding.get_parameters());

            const rp = params['dynamics']['rotor_positions'];
            const k_m = params['dynamics']['rotor_torque_constants'];
            const k_m_dir = params['dynamics']['rotor_torque_directions'].map(v => v[2]); // only considering the z direction
            const [a, b, c] = params['dynamics']['rotor_thrust_coefficients'][0];

            const omega = vec3.fromValues(...state['angular_velocity']);

            const omega_range = [1.5, 1.5, -1]; // yaw command is clockwise
            const max_thrust = params["dynamics"]["rotor_thrust_coefficients"].map(tc => tc.reduce((a, c, i) => a + c * Math.pow(params.dynamics.action_limit.max, i), 0)).reduce((a, c) => a + c, 0);
            const omega_des_array = this.actions !== null ? ["roll", "pitch", "yaw"].map((k, i) => this.actions[k] * omega_range[i]) : [0, 0, 0];
            const omega_des = vec3.fromValues(...omega_des_array);

            const omega_error = vec3.subtract(vec3.create(), omega, omega_des);
            const tau = vec3.create();
            vec3.scale(tau, omega_error, -this.k_omega);

            const T = this.actions !== null ? (this.actions["thrust"] + 1) / 2 * max_thrust : 0;

            const controlInputs = vec4.fromValues(T, tau[0], tau[1], tau[2]);

            const A = mat4.transpose(mat4.create(), mat4.fromValues(
                1,         1,         1,         1,
                +rp[0][1], +rp[1][1], +rp[2][1], +rp[3][1], // positive y rotor displacement causes positive x/roll 
                -rp[0][0], -rp[1][0], -rp[2][0], -rp[3][0], // positive x rotor displacement causes negative y/pitch
                k_m_dir[0]*k_m[0], k_m_dir[1]*k_m[1], k_m_dir[2]*k_m[2], k_m_dir[3]*k_m[3]
            ));
            const A_inv = mat4.invert(mat4.create(), A);

            const f = vec4.transformMat4(vec4.create(), controlInputs, A_inv);
            const f_clipped = f.map(fi => Math.max(0, fi));

            function solveRPM(fi) {
                const discriminant = b * b - 4 * a * (c - fi);
                if (discriminant < 0 || fi < c) return 0.0;
                const rpm = (-b + Math.sqrt(discriminant)) / (2 * a);
                return Math.max(0.0, Math.min(1.0, rpm));
            }
            
            function solveRPM(fi){
                // a + b*rpm + c*rpm^2 = fi
                // c*rpm^2 + b*rpm + (a - fi) = 0
                const b_new = b/c
                const a_new = (a - fi)/c
                const d = b_new*b_new/4 - a_new
                if (d < 0) return 0.0;
                let rpm = -b_new/2 + Math.sqrt(d);
                if (rpm < 0){
                    rpm = -b_new/2 - Math.sqrt(d);
                };
                return rpm
            }

            const rpms = f_clipped.map(solveRPM);
            const rpms_clipped = rpms.map(x=> x > 1 ? 1 : (x < -1 ? -1 : x))
            return rpms_clipped.map(x => x * 2 - 1)
        })
    }
    get_reference(states){
        return null
    }
    reset() {
    }
}
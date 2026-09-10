import { vec3, mat3, vec4, mat4, quat } from 'gl-matrix'; // vendored through the importmap in index.html

export default class Controller {
    constructor() {
        this.k_p = 5.0;
        this.k_d = 5;
        this.k_R = 0.001;
        this.k_omega = 0.001;
    }

    evaluate_step(state) {
        const stateDict = JSON.parse(state.get_state());
        const params = JSON.parse(state.get_parameters());

        const p = vec3.fromValues(...stateDict['position']);
        const q = quat.fromValues(
            stateDict['orientation'][1],
            stateDict['orientation'][2],
            stateDict['orientation'][3],
            stateDict['orientation'][0]
        );
        const v = vec3.fromValues(...stateDict['linear_velocity']);
        const omega = vec3.fromValues(...stateDict['angular_velocity']);

        const m = params['dynamics']['mass'];
        const g = -params['dynamics']['gravity'][2];
        const rp = params['dynamics']['rotor_positions'];
        const k_m = params['dynamics']['rotor_torque_constants'];
        const k_m_dir = params['dynamics']['rotor_torque_directions'].map(v => v[2]); // only considering the z direction
        const [a, b, c] = params['dynamics']['rotor_thrust_coefficients'][0];

        const p_des = vec3.fromValues(0, 0, 0);
        const v_des = vec3.fromValues(0, 0, 0);
        const q_des = quat.fromValues(0, 0, 0, 1);

        const e_p = vec3.subtract(vec3.create(), p, p_des);
        const e_v = vec3.subtract(vec3.create(), v, v_des);

        const a_fb = vec3.add(
            vec3.create(),
            vec3.scale(vec3.create(), e_p, -this.k_p),
            vec3.scale(vec3.create(), e_v, -this.k_d)
        );

        const F_des = vec3.fromValues(0, 0, m * g);
        vec3.add(F_des, F_des, vec3.scale(vec3.create(), a_fb, m));

        // note: gl-matrix uses a column-major layout
        const A = mat4.transpose(mat4.create(), mat4.fromValues(
            1,         1,         1,         1,
            +rp[0][1], +rp[1][1], +rp[2][1], +rp[3][1], // positive y rotor displacement causes positive x/roll 
            -rp[0][0], -rp[1][0], -rp[2][0], -rp[3][0], // positive x rotor displacement causes negative y/pitch
            k_m_dir[0]*k_m[0], k_m_dir[1]*k_m[1], k_m_dir[2]*k_m[2], k_m_dir[3]*k_m[3]
        ));
        const A_inv = mat4.invert(mat4.create(), A);

        const T = vec3.dot(F_des, vec3.transformQuat(vec3.create(), [0, 0, 1], q));

        const R = mat3.fromQuat(mat3.create(), q);
        const R_des_z = vec3.normalize(vec3.create(), F_des)
        const R_des_y = vec3.cross(vec3.create(), R_des_z, vec3.fromValues(1, 0, 0));
        const R_des_x = vec3.cross(vec3.create(), R_des_y, R_des_z);
        const R_des = mat3.fromValues(
            R_des_x[0], R_des_x[1], R_des_x[2],
            R_des_y[0], R_des_y[1], R_des_y[2],
            R_des_z[0], R_des_z[1], R_des_z[2]
        );

        const R_err = mat3.create();
        mat3.multiply(R_err, mat3.transpose(mat3.create(), R_des), R);
        mat3.subtract(R_err, R_err, mat3.multiply(mat3.create(), mat3.transpose(mat3.create(), R), R_des));

        const e_R = vec3.fromValues(
            R_err[5], -R_err[2], R_err[1]
        );

        const tau = vec3.create();
        vec3.scale(tau, e_R, -this.k_R);
        const temp = vec3.create();
        vec3.scale(temp, omega, -this.k_omega);
        vec3.add(tau, tau, temp);

        const controlInputs = vec4.fromValues(T, tau[0], tau[1], tau[2]);
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
    }

    reset() { }
}

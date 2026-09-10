// Mapping from platform names to conta mesh hashes
const PLATFORM_MESH_MAP = {
    "x500": "9602ffc2ffb77f62c4cf6fdc78fe67d32088870d",
    "crazyflie": "b75f5120e17783744a8fac5e1ab69c2dce10f0e3",
    "arpl": "775ba8559aeed800dbcdab93806601e39d84fede"
}

function addMeshToParameters(parameters, platform) {
    if (!parameters.ui && PLATFORM_MESH_MAP[platform]) {
        parameters.ui = {
            enable: true,
            model: PLATFORM_MESH_MAP[platform]
        }
    }
    return parameters
}

export class SimControls{
    constructor(l2f, policy){
        this.policy = policy
        const num_vehicles_input = document.getElementById("num-vehicles")
        num_vehicles_input.addEventListener("input", async () => {
            let platform = document.getElementById("vehicle-load-dynamics-selector").value
            platform = platform === "file" ? "crazyflie" : platform
            const parameters = await (await fetch(`./blob/registry/${platform}.json`)).json()
            addMeshToParameters(parameters, platform)
            const diff = await l2f.change_num_quadrotors(parseInt(num_vehicles_input.value), parameters)
        })

        const speed_slider = document.getElementById("speed-slider")
        const speed_info = document.getElementById("speed-info")
        const speed_info_real = document.getElementById("speed-info-real")
        speed_slider.addEventListener("input", () => {
            const normalized_value = parseFloat(speed_slider.value)
            const range = 9;
            const factor = normalized_value > 0.5 ? (1 + (normalized_value - 0.5) * 2 * range) : (1 + (0.5 - normalized_value) * 2 * range)
            l2f.speed = normalized_value > 0.5 ? factor : 1 / factor
            speed_info.innerText = `Speed: ${normalized_value > 0.5 ? factor.toFixed(1) : (normalized_value === 0.5 ? "1" : `1/${factor.toFixed(1)}`)}x`
        })
        setInterval(() => {
            const f = l2f.real_time_factor
            if(f){
                speed_info_real.innerText = `(${f > 1 ? f.toFixed(1) : `1/${(1/f).toFixed(1)}`}x)`
            }
            else{
                speed_info_real.innerText = `(-)`
            }
        }, 100)

        const show_axes_checkbox = document.getElementById("show-axes")
        show_axes_checkbox.addEventListener("change", () => {
            l2f.initialized.then(() => {
                l2f.ui_state.showAxes = show_axes_checkbox.checked
                l2f.ui.episode_init_multi(l2f.ui_state, l2f.parameters)
            })
        })

        const pause_on_reset_checkbox = document.getElementById("pause-on-reset")

        const sample_or_not = (state, sample) => {
            if(sample){
                state.sample_initial_state()
            }
            else{
                state.initial_state()
            }
        }
        document.getElementById("sample-initial-states").addEventListener("click", () => {
            l2f.states.forEach(state => {
                if(pause_on_reset_checkbox.checked && pause_button.innerText === "Pause"){
                    pause_button.innerText = "Resume"
                    l2f.request_pause = true
                }
                sample_or_not(state, true)
            })
            this.policy.reset()
            l2f.resetTrackingError()
        })
        document.getElementById("initial-states").addEventListener("click", () => {
            l2f.states.forEach(state => {
                if(pause_on_reset_checkbox.checked && pause_button.innerText === "Pause"){
                    pause_button.innerText = "Resume"
                    l2f.request_pause = true
                }
                sample_or_not(state, false)
            })
            this.policy.reset()
            l2f.resetTrackingError()
        })
        const pause_button = document.getElementById("pause")
        pause_button.addEventListener("click", () => {
            if(pause_button.innerText === "Pause"){
                pause_button.innerText = "Resume"
                l2f.request_pause = true
            }
            else{
                pause_button.innerText = "Pause"
                l2f.request_unpause = true
            }
        })
        const step_button = document.getElementById("step")
        step_button.addEventListener("click", () => {
            if(pause_button.innerText === "Pause"){
                pause_button.innerText = "Resume"
                l2f.request_pause = true
            }
            l2f.simulate_step()
        })
    }

}
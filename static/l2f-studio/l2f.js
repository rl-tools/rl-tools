// import * as ui from "./ui.js"
import createModule from "l2f-interface";
const DEBUG = false

export class L2F{
    constructor(parent, num_quadrotors, policy, seed){
        this.canvas = document.createElement('canvas');
        if(DEBUG){
            this.canvas.style.backgroundColor = "white"
        }
        const dpr = window.devicePixelRatio || 1;
        const resizeCanvas = () => {
            const parentRect = parent.getBoundingClientRect();
            this.canvas.style.width = parentRect.width + 'px';
            this.canvas.style.height = parentRect.height + 'px';
            this.canvas.width = parentRect.width * dpr;
            this.canvas.height = parentRect.height * dpr;
        };
        resizeCanvas()
        window.addEventListener('resize', resizeCanvas.bind(this), false);
        this.policy = policy

        this.initialized = createModule().then(async (l2f_interface) => {
            this.states = [...Array(num_quadrotors)].map((_, i) =>new l2f_interface.State(seed + i));
            if(DEBUG){
                this.ui = ui
            }
            else{
                const blob = new Blob([this.states[0].get_ui()], { type: 'application/javascript' });
                const url = URL.createObjectURL(blob);
                this.ui = await import(url)
                URL.revokeObjectURL(url);
            }
            this.ui_state = await this.ui.init(this.canvas, {devicePixelRatio: window.devicePixelRatio})
            if(this.ui_state.cursor_grab){
                this.canvas.style.cursor = "grab"
            }
            parent.appendChild(this.canvas);
            this.parameters = this.states.map(state => JSON.parse(state.get_parameters()))
            await this.ui.episode_init_multi(this.ui_state, this.parameters)
            this.render()
        });
        this.last_step = null
        this.last_dt = 0
    }
    async render(){
        const now = performance.now()
        if(this.last_step === null || (now - this.last_step) / 1000 > this.last_dt){
            this.last_step = now

            this.states.forEach(state => {
                const action = this.policy(state)
                console.assert(action.length === state.action_dim, "Action dimension mismatch")
                action.map((v, i) => {
                    state.set_action(i, v)
                })
                this.last_dt = state.step()
            })
        }
        const current_states =  this.states.map(state => JSON.parse(state.get_state()))
        const current_actions = this.states.map(state => JSON.parse(state.get_action()))
        await this.ui.render_multi(this.ui_state, this.parameters, current_states, current_actions)
        requestAnimationFrame(() => this.render());
    }
}

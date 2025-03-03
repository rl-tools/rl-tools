// import * as ui from "./ui.js"
import createModule from "l2f-interface";
// const DEBUG = true
const DEBUG = false

import Stats from 'https://esm.sh/stats.js'

export class L2F{
    constructor(parent, num_quadrotors, policy, seed){

        this.seed = seed

        const urlParams = new URLSearchParams(window.location.search);
        this.DEBUG = urlParams.has('DEBUG') ? urlParams.get('DEBUG') === 'true' : false


        if(this.DEBUG){
            this.stats = new Stats();
            this.stats.showPanel(0);

            this.stats.dom.style.transform = 'scale(3)';
            this.stats.dom.style.transformOrigin = 'top left';
            this.stats.dom.style.left = '0px';
            this.stats.dom.style.top = '0px';
            this.stats.dom.style.position = 'fixed';
            document.body.appendChild(this.stats.dom);
        }

        this.overtimes = []
        this.control_tick = 0

        this.pause = false
        this.speed = 1
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
            this.l2f_interface = l2f_interface
            this.states = [...Array(num_quadrotors)].map((_, i) =>new this.l2f_interface.State(this.seed + i));
            this.parameters = this.states.map(state => JSON.parse(state.get_parameters()))
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
            await this.ui.episode_init_multi(this.ui_state, this.parameters)

            this.render()
            setInterval(this.control.bind(this), 0)
        });
        this.last_step = null
        this.last_dt = 0
    }
    async change_num_quadrotors(num){
        const diff = num - this.states.length
        if(diff > 0){
            const new_states = [...Array(num - this.states.length)].map((_, i) =>new this.l2f_interface.State(this.seed + this.states.length + i));
            this.states = this.states.concat(new_states)
        }
        else{
            this.states = this.states.slice(0, num)
        }
        this.parameters = this.states.map(state => JSON.parse(state.get_parameters()))
        await this.ui.episode_init_multi(this.ui_state, this.parameters)
        return diff
    }
    simulate_step(){
        this.states.forEach(state => {
            const action = this.policy(state)
            console.assert(action.length === state.action_dim, "Action dimension mismatch")
            action.map((v, i) => {
                state.set_action(i, v)
            })
            this.last_dt = state.step()
        })
    }

    async control(){
        const now = performance.now()
        if(!this.pause && (this.last_step === null || (now - this.last_step) / 1000 > this.last_dt / this.speed)){
            const overtime = (now - this.last_step) / 1000 - this.last_dt / this.speed
            this.overtimes.push(overtime)
            this.overtimes = this.overtimes.slice(-10)
            if(this.control_tick % 100 === 0){
                // console.log(`Average overtime: ${this.overtimes.reduce((a, b) => a + b, 0) / this.overtimes.length}`)
            }
            this.last_step = now
            this.simulate_step()
        }
        this.control_tick += 1
    }
    async render(){
        if(this.DEBUG){
            this.stats.begin()
        }
        const current_states =  this.states.map(state => JSON.parse(state.get_state()))
        const current_actions = this.states.map(state => JSON.parse(state.get_action()))
        await this.ui.render_multi(this.ui_state, this.parameters, current_states, current_actions)
        if(this.DEBUG){
            this.stats.end()
        }
        requestAnimationFrame(() => this.render());
    }
}

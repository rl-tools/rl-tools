class Mapper{
    constructor(type, name, gamepad, control_map, completion_handler){
        this.type = type
        this.name = name
        this.active = false
        this.index = -1
        this.live_index = -1
        this.max_deflection = 0
        this.base = null
        this.invert = false
        this.base_values = {}
        this.control_map = control_map
        for (let i = 0; i < gamepad.axes.length; i++) {
            this.base_values[i] = gamepad.axes[i];
        }
        this.finished = false
        this.completion_handler = completion_handler
    }
    handle_mapping_axis(gamepad){
        if (!this.active) {
            let maxDiff = 0;
            for (let i = 0; i < gamepad.axes.length; i++) {
                const diff = Math.abs(gamepad.axes[i] - this.base_values[i]);
                if (diff > maxDiff) {
                    maxDiff = diff;
                    this.live_index = i;
                    this.invert = (gamepad.axes[i] - this.base_values[i]) < 0;
                    this.base = this.base_values[i];
                }
                if (diff > 0.5) {
                    this.base = this.base_values[i];
                    this.active = true;
                    this.index = i;
                    const movement = gamepad.axes[i] - this.base_values[i];
                    this.invert = movement < 0;
                    break;
                }
            }
        } else {
            const value = gamepad.axes[this.index];
            const deflection = Math.abs(value - this.base);
            if (deflection > this.max_deflection) {
                this.max_deflection = deflection;
            } 
            if (this.max_deflection - deflection > 0.1){
                this.finished = true
            }
        }
        // live_view
        const was_not_mapped = this.control_map[this.name] === undefined;
        this.control_map[this.name] = { 
            type: 'axis', 
            index: this.active ? this.index : this.live_index, 
            invert: this.invert,
            base: this.base,
            max_deflection: this.max_deflection,
            finished: this.finished,
            expo: 0
        };
        if(this.finished){
            this.completion_handler();
        }
        return was_not_mapped
    }
    handle_mapping_button(gamepad){
        for (let i = 0; i < gamepad.buttons.length; i++) {
            if (gamepad.buttons[i].pressed) {
                this.control_map[this.name] = { type: 'button', index: i};
                this.completion_handler()
                return false;
            }
        }
        return false
    }
    handle_mapping(gamepad){
        if(this.type === 'axis'){
            return this.handle_mapping_axis(gamepad);
        } else if(this.type === 'button'){
            return this.handle_mapping_button(gamepad);
        }
    }
}
export class Gamepad{
    constructor(element, gamepad_interface, options = {}){
        this.element = element;
        this.gamepad_interface = gamepad_interface;
        this.storage_key = options.storageKey || "gamepad_config";
        this.enabled = options.enabled ?? true;
        this.destroyed = false;
        this.mapper = null;
        this.control_map = {}
        this.callbacks = {}
        this.expo_sliders = {}
        this.gamepad_index = null
        this.gamepad_poller = null
        this.status_element = this.element.querySelector('.gamepad-status, #gamepad-status');
        this.controls_element = this.element.querySelector('.gamepad-controls, #gamepad-controls');
        this.state_element = this.element.querySelector('.gamepad-state, #gamepad-state');
        this.button_container = this.element.querySelector('.gamepad-button-container, #gamepad-button-container');
        this.axis_template = this.element.querySelector('.gamepad-axis-template, #gamepad-axis-template') || document.getElementById('gamepad-axis-template');
        this.button_template = this.element.querySelector('.gamepad-button-template, #gamepad-button-template') || document.getElementById('gamepad-button-template');
        if(!this.status_element || !this.state_element || !this.button_container || !this.axis_template || !this.button_template){
            throw new Error("Gamepad widget is missing required DOM elements");
        }
        this.expo_curve = (x, expo) => {
            return (1-expo)*x + expo * Math.pow(x, 3)
        }
        const reset_button = document.createElement('button');
        reset_button.classList.add('gamepad-mapping-button');
        reset_button.classList.add('gamepad-button');
        reset_button.classList.add('gamepad-reset-button')
        reset_button.textContent = "Reset";
        reset_button.onclick = () => {
            const gamepad = this.get_gamepad();
            if(gamepad === null) return;
            this.gamepad_index = null
            this.control_map = {}
            this.callbacks = {}
            this.status_element.textContent = 'No gamepad connected';
            this.status_element.className = 'gamepad-status gamepad-disconnected';
            this.reset_config(gamepad.id)
            this.set_mapping_buttons_enabled(false);
            this.render_live_view()
        };
        this.button_container.innerHTML = '';
        this.button_container.appendChild(reset_button);


        for (const channel in gamepad_interface){
            const details = gamepad_interface[channel];
            const button = document.createElement('button');
            // <button class="gamepad-mapping-button" disabled onclick="startMapping('thrust', 'axis', 'up', this)">Map Thrust Axis</button>
            button.classList.add('gamepad-mapping-button');
            button.classList.add('gamepad-button');
            const default_text = `Map ${channel}`
            button.textContent = default_text;
            button.disabled = true;
            button.onclick = () => {
                if(!this.enabled) return;
                button.textContent = details.type === "button" ? 'Press Button' : `Move ${details.positive_direction}`;
                this.set_mapping_buttons_enabled(false);
                button.disabled = false;
                const gamepad = this.get_gamepad();
                if(this.mapper === null && gamepad !== null){
                    this.mapper = new Mapper(details.type, channel, gamepad, this.control_map, () => {
                        button.textContent = default_text
                        this.set_mapping_buttons_enabled(true);
                        this.mapper = null
                        this.save_config(gamepad.id)
                        this.render_live_view()
                    });
                }
            };
            this.button_container.appendChild(button);
        }
        this.listeners = []
        this.setEnabled(this.enabled)
        this.poll()
    }
    set_mapping_buttons_enabled(enabled){
        this.element.querySelectorAll('.gamepad-mapping-button').forEach((btn) => { btn.disabled = !enabled; });
    }
    setEnabled(enabled){
        this.enabled = enabled;
        this.mapper = null;
        this.set_mapping_buttons_enabled(Boolean(enabled && this.gamepad_index !== null));
        if(!enabled){
            this.status_element.textContent = 'Gamepad input inactive';
            this.status_element.className = 'gamepad-status gamepad-disconnected';
        }
        else if(this.gamepad_index === null){
            this.status_element.textContent = 'No gamepad connected (please press button or move axis to connect)';
            this.status_element.className = 'gamepad-status gamepad-disconnected';
        }
        else{
            const gamepad = navigator.getGamepads ? navigator.getGamepads()[this.gamepad_index] : null;
            if(gamepad){
                this.status_element.textContent = 'Gamepad connected: ' + gamepad.id;
                this.status_element.className = 'gamepad-status gamepad-connected';
            }
            else{
                this.gamepad_index = null;
                this.status_element.textContent = 'No gamepad connected (please press button or move axis to connect)';
                this.status_element.className = 'gamepad-status gamepad-disconnected';
                this.set_mapping_buttons_enabled(false);
            }
        }
    }
    destroy(){
        this.destroyed = true;
        this.listeners = [];
        this.mapper = null;
    }
    load_config(id){
        let gamepad_config = localStorage.getItem(this.storage_key);
        gamepad_config = gamepad_config !== null ? JSON.parse(gamepad_config) : {};
        gamepad_config = id in gamepad_config ? gamepad_config[id] : null;
        if(gamepad_config !== null){
            this.control_map = gamepad_config;
            this.render_live_view()
        }
    }
    reset_config(id){
        let gamepad_config = localStorage.getItem(this.storage_key);
        gamepad_config = gamepad_config !== null ? JSON.parse(gamepad_config) : null;
        if(gamepad_config !== null){
            delete gamepad_config[id];
            localStorage.setItem(this.storage_key, JSON.stringify(gamepad_config));
        }
    }
    save_config(id){
        let gamepad_config = localStorage.getItem(this.storage_key);
        gamepad_config = gamepad_config !== null ? JSON.parse(gamepad_config) : {};
        gamepad_config[id] = this.control_map;
        localStorage.setItem(this.storage_key, JSON.stringify(gamepad_config));
    }
    get_gamepad(){
        if(!this.enabled) return null;
        if(this.gamepad_index === null){
            const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
            for (let i = 0; i < gamepads.length; i++) {
                const gp = gamepads[i];
                if(gp){
                    this.gamepad_index = gp.index;
                    this.status_element.textContent = 'Gamepad connected: ' + gp.id;
                    this.status_element.className = 'gamepad-status gamepad-connected';
                    this.set_mapping_buttons_enabled(true);
                    this.load_config(gp.id);
                    break;
                }
            }
        }
        return this.gamepad_index !== null ? navigator.getGamepads()[this.gamepad_index] : null;
    }
    render_live_view(){
        this.state_element.innerHTML = '';
        this.callbacks = {}
        this.expo_sliders = {}
        for (const control in this.control_map){
            const details = this.control_map[control];
            const template = details.type === 'axis' ? this.axis_template : this.button_template;
            const clone = template.content.cloneNode(true);
            this.state_element.appendChild(clone);
            const name = this.state_element.lastElementChild.querySelector('.gamepad-controls-name');
            name.textContent = control
            const element = this.state_element.lastElementChild
            let expo_plot = null
            if(details.type === 'axis'){
                const expo_canvas = element.querySelector('.gamepad-expo-canvas');
                const expo_slider = element.querySelector('.gamepad-slider-expo')
                this.expo_sliders[control] = expo_slider;
                if(expo_slider){
                    expo_slider.value = this.control_map[control].expo ?? 0
                    expo_slider.addEventListener('input', (event) => {
                        this.control_map[control].expo = event.target.value;
                        const gamepad = this.get_gamepad();
                        if(gamepad) this.save_config(gamepad.id);
                    })
                }
                if(expo_canvas){
                    expo_plot = new ExpoPlot(expo_canvas, this.expo_curve);
                }
            }
            this.callbacks[control] = (value_raw) =>{
                const slider = element.querySelector('.gamepad-slider');
                const valueDisplay = element.querySelector('.gamepad-value-display');
                if (details.type === 'axis') {
                    if(slider) slider.value = value_raw;
                    const expo = this.expo_sliders[control]?.value ?? 0;
                    const processed_value = this.expo_curve(value_raw, expo);
                    valueDisplay.textContent = processed_value.toFixed(2);
                    if(expo_plot) expo_plot.draw(value_raw, expo);
                } else {
                    const buttonIndicator = element.querySelector('.gamepad-button-indicator');
                    buttonIndicator.style.backgroundColor = value_raw ? '#28a745' : '#ccc';
                    valueDisplay.textContent = value_raw ? 'Pressed' : 'Released';
                }
            };
        }
    }
    poll(){
        if(this.destroyed) return;
        requestAnimationFrame(this.poll.bind(this));
        if(!this.enabled) return;
        const gamepad = this.get_gamepad();
        if(gamepad){
            if(this.mapper){
                const should_render = this.mapper.handle_mapping(gamepad);
                if(should_render){
                    this.render_live_view();
                }
            }
            const output = {}
            for(const control in this.control_map){
                const details = this.control_map[control];
                if(details.index === -1) continue;
                const raw_value = gamepad[details.type === 'axis' ? 'axes' : 'buttons'][details.index];
                if(details.type === 'axis'){
                    const deflection = Number(details.max_deflection);
                    const scale = Number.isFinite(deflection) && deflection > 0 ? deflection : 1;
                    const value = (raw_value - details.base) / scale;
                    const value_clipped = Math.max(-1, Math.min(1, value));
                    const value_inverted = details.invert ? -value_clipped : value_clipped;
                    const value_transformed = this.expo_curve(value_inverted, this.expo_sliders[control]?.value ?? details.expo ?? 0);
                    output[control] = value_transformed;
                    if(this.callbacks[control]) this.callbacks[control](value_inverted);
                }
                else{
                    output[control] = raw_value.pressed;
                    if(this.callbacks[control]) this.callbacks[control](raw_value.pressed);
                }
            }
            if((Object.keys(this.gamepad_interface).every((key) => key in output && this.control_map[key].index !== -1))){
                for(const listener of this.listeners){
                    listener(output);
                }
            }
        }
    }
    addListener(listener){
        this.listeners.push(listener);
    }
}


class ExpoPlot{
    constructor(canvas, mapping){
        this.canvas = canvas
        this.mapping = mapping
        const ctx = canvas.getContext("2d");
    }
    draw(x, expo) {
        const ctx = this.canvas.getContext("2d");
        const w = this.canvas.width;
        const h = this.canvas.height;

        ctx.clearRect(0, 0, w, h);

        // Draw axes
        ctx.strokeStyle = "#ccc";
        ctx.beginPath();
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.moveTo(w / 2, 0);
        ctx.lineTo(w / 2, h);
        ctx.stroke();

        // Draw expo curve
        ctx.strokeStyle = "blue";
        ctx.beginPath();
        for (let i = 0; i <= w; i++) {
            const x_norm = (i / w) * 2 - 1; // normalize to [-1, 1]
            const y_val = this.mapping(x_norm, expo);
            const y_pix = h / 2 - y_val * (h / 2);
            if (i === 0) ctx.moveTo(i, y_pix);
            else ctx.lineTo(i, y_pix);
        }
        ctx.stroke();

        // Draw current position dot and projections
        const x_pix = w / 2 + x * (w / 2);
        const y_val = this.mapping(x, expo);
        const y_pix = h / 2 - y_val * (h / 2);

        ctx.strokeStyle = "#999";
        ctx.setLineDash([4, 2]);
        ctx.beginPath();
        ctx.moveTo(x_pix, h / 2);
        ctx.lineTo(x_pix, y_pix);
        ctx.moveTo(w / 2, y_pix);
        ctx.lineTo(x_pix, y_pix);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = "red";
        ctx.beginPath();
        ctx.arc(x_pix, y_pix, 4, 0, 2 * Math.PI);
        ctx.fill();
    }
}

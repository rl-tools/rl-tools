const HISTORY = 160
const HEIGHT = 24
const ZERO_LINE_COLOR = "#ddd"
const RESET_COLOR = "#bbb"
export const ACTION_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231"]

export class ActionSparklines{
    constructor(container){
        this.container = container
        this.vehicles = []
        this.dirty = false
        this.frame_requested = false
        this.draw = this.draw.bind(this)
        window.addEventListener("resize", () => this.request_frame())
    }
    attach(canvas, vehicle_id, action_dim){
        const dpr = window.devicePixelRatio || 1
        canvas.width = HISTORY * dpr
        canvas.height = HEIGHT * dpr
        canvas.style.width = `${HISTORY}px`
        canvas.style.height = `${HEIGHT}px`
        const ctx = canvas.getContext("2d")
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
        this.vehicles[vehicle_id] = {
            ctx: ctx,
            dim: action_dim,
            values: new Float32Array(HISTORY * action_dim),
            resets: new Uint8Array(HISTORY),
            head: 0,
            count: 0,
            last_steps: -1
        }
        this.vehicles.length = Math.max(this.vehicles.length, vehicle_id + 1)
        this.dirty = true
    }
    record(states){
        const n = Math.min(states.length, this.vehicles.length)
        for(let i = 0; i < n; i++){
            const vehicle = this.vehicles[i]
            const steps = states[i].stepsSinceReset
            if(vehicle === undefined || steps === vehicle.last_steps){
                continue
            }
            const action = states[i].action
            const offset = vehicle.head * vehicle.dim
            for(let j = 0; j < vehicle.dim; j++){
                vehicle.values[offset + j] = action[j]
            }
            vehicle.resets[vehicle.head] = vehicle.count > 0 && steps !== vehicle.last_steps + 1 ? 1 : 0
            vehicle.last_steps = steps
            vehicle.head = vehicle.head + 1 === HISTORY ? 0 : vehicle.head + 1
            if(vehicle.count < HISTORY){
                vehicle.count += 1
            }
            this.dirty = true
        }
        this.request_frame()
    }
    request_frame(){
        if(this.frame_requested || !this.dirty){
            return
        }
        this.frame_requested = true
        requestAnimationFrame(this.draw)
    }
    draw(){
        this.frame_requested = false
        if(!this.dirty || this.container.clientWidth === 0){
            return
        }
        this.dirty = false
        const mid = HEIGHT / 2
        const scale = mid - 1
        for(const vehicle of this.vehicles){
            if(vehicle === undefined || vehicle.count === 0){
                continue
            }
            const ctx = vehicle.ctx
            const start = (vehicle.head - vehicle.count + HISTORY) % HISTORY
            const x_offset = HISTORY - vehicle.count
            ctx.clearRect(0, 0, HISTORY, HEIGHT)
            ctx.strokeStyle = ZERO_LINE_COLOR
            ctx.beginPath()
            ctx.moveTo(0, mid)
            ctx.lineTo(HISTORY, mid)
            ctx.stroke()
            ctx.strokeStyle = RESET_COLOR
            ctx.beginPath()
            for(let k = 0, idx = start; k < vehicle.count; k++){
                if(vehicle.resets[idx]){
                    ctx.moveTo(x_offset + k, 0)
                    ctx.lineTo(x_offset + k, HEIGHT)
                }
                idx = idx + 1 === HISTORY ? 0 : idx + 1
            }
            ctx.stroke()
            for(let j = 0; j < vehicle.dim; j++){
                ctx.strokeStyle = ACTION_COLORS[j % ACTION_COLORS.length]
                ctx.beginPath()
                for(let k = 0, idx = start; k < vehicle.count; k++){
                    const value = Math.max(-1, Math.min(1, vehicle.values[idx * vehicle.dim + j]))
                    const y = mid - value * scale
                    if(k === 0){
                        ctx.moveTo(x_offset + k, y)
                    }
                    else{
                        ctx.lineTo(x_offset + k, y)
                    }
                    idx = idx + 1 === HISTORY ? 0 : idx + 1
                }
                ctx.stroke()
            }
        }
    }
}

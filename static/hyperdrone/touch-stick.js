export function create_touch_stick(element, enabled){
    const thumb = element.querySelector(".stick-thumb");
    let pointer_id = null;
    let center_x = 0;
    let center_y = 0;
    let radius = 0;
    const stick = {x: 0, y: 0, clear};

    function clear(){
        const released_id = pointer_id;
        pointer_id = null;
        stick.x = 0;
        stick.y = 0;
        thumb.style.setProperty("--stick-x", "0px");
        thumb.style.setProperty("--stick-y", "0px");
        if(released_id !== null && element.hasPointerCapture(released_id)){
            element.releasePointerCapture(released_id);
        }
    }

    function update(event){
        const x = (event.clientX - center_x) / radius;
        const y = (event.clientY - center_y) / radius;
        const distance = Math.hypot(x, y);
        const clamped = Math.min(1, distance);
        const scale = Math.max(0, (clamped - 0.1) / 0.9) / (distance || 1);
        stick.x = x * scale;
        stick.y = y * scale;
        const visual_scale = clamped * radius / (distance || 1);
        thumb.style.setProperty("--stick-x", `${x * visual_scale}px`);
        thumb.style.setProperty("--stick-y", `${y * visual_scale}px`);
    }

    element.addEventListener("pointerdown", event => {
        if(!enabled() || pointer_id !== null || event.button !== 0){
            return;
        }
        const bounds = element.getBoundingClientRect();
        const thumb_bounds = thumb.getBoundingClientRect();
        radius = (Math.min(bounds.width, bounds.height) - Math.max(thumb_bounds.width, thumb_bounds.height)) / 2;
        if(radius <= 0){
            return;
        }
        center_x = bounds.left + bounds.width / 2;
        center_y = bounds.top + bounds.height / 2;
        pointer_id = event.pointerId;
        element.setPointerCapture(pointer_id);
        event.preventDefault();
        update(event);
    });
    element.addEventListener("pointermove", event => {
        if(event.pointerId !== pointer_id){
            return;
        }
        event.preventDefault();
        if(!enabled()){
            clear();
            return;
        }
        update(event);
    });
    for(const type of ["pointerup", "pointercancel", "lostpointercapture"]){
        element.addEventListener(type, event => {
            if(event.pointerId === pointer_id){
                event.preventDefault();
                clear();
            }
        });
    }
    element.addEventListener("contextmenu", event => event.preventDefault());
    clear();
    return stick;
}

function mean(a){
    return a.reduce((sum, val) => sum + val, 0) / a.length
}
function std(x){
    const sample_mean = mean(x)
    const variance = x.reduce((sum, val) => sum + (val - sample_mean) ** 2, 0)
    return variance > 0 ?  Math.sqrt(variance / x.length) : 0;
}

export class ParameterManager{
    constructor(l2f){
        this.l2f = l2f
        this.initialized = new Promise((resolve, reject) => {
            this.initialized_resolve = resolve
        })
        this.l2f.initialized.then(() => {
            const perturbation_id_input = document.getElementById("perturbation-id-input")
            const perturbation_id_status = document.getElementById("perturbation-id-status")
            const perturbation_id_min = document.getElementById("perturbation-id-min")
            const perturbation_id_max = document.getElementById("perturbation-id-max")
            const perturbation_transform = document.getElementById("perturbation-transform")
            const perturbation_id_suggestion = document.getElementById("perturbation-id-suggestion")
            const perturbation_groups = document.getElementById("perturbation-groups")
            const perturbation_group_template = document.getElementById("perturbation-group-template")
            perturbation_id_input.addEventListener("input", () => {
                const parent_path = perturbation_id_input.value.split('.').slice(0, -1).join('.')
                const found_values = this.get_values_from_path(perturbation_id_input.value, this.l2f.parameters)
                console.log("found_values: ", found_values)
                let parent = this.get_values_from_path(parent_path, this.l2f.parameters)
                if(parent !== undefined && (typeof parent[0] === "object")){
                    let current_key = perturbation_id_input.value.split('.').slice(-1)[0]
                    let foresight = false
                    if(current_key in parent[0]){
                        parent = parent.map(x => x[current_key])
                        foresight = true
                        current_key = ""
                    }
                    const suggestions = Object.keys(parent[0]).sort().reduce((acc, key) => {
                        if(key.startsWith(current_key)){
                            acc.push(key)
                        }
                        return acc
                    }, [])
                    if(suggestions.length > 0){
                        const marked_suggestions = suggestions.map(x => {
                            return (foresight ? "." : "") + "<b>" + current_key + "</b>" + x.slice(current_key.length)
                        })
                        perturbation_id_suggestion.innerHTML = "Subgroups: " + marked_suggestions.join(", ")
                    }
                    else{
                        if(found_values === undefined){
                            perturbation_id_suggestion.innerText = "Subgroups: " + "not found"
                        }
                        else{
                            if(found_values.length > 3){
                                perturbation_id_suggestion.innerText = `Values (${found_values.length}): ${mean(found_values).toExponential(2)} ± ${std(found_values).toExponential(2)}`;
                            }
                        }
                    }
                }
                if(found_values !== undefined && (typeof found_values[0] === "number")){
                    perturbation_id_status.innerText = "✅"
                    const range = 2
                    const found_negative = found_values.some(x => x < 0)
                    const found_positive = found_values.some(x => x > 0)
                    if(found_positive && found_negative){
                        perturbation_id_min.value = Math.min(...found_values) * range;
                        perturbation_id_max.value = Math.max(...found_values) * range;
                    }
                    else{
                        if(!found_negative && !found_positive){
                            perturbation_id_min.value = 0;
                            perturbation_id_max.value = 0;
                        }
                        else{
                            perturbation_id_min.value = found_positive ? Math.min(...found_values) / range : Math.min(...found_values) * range;
                            perturbation_id_max.value = found_positive ? Math.max(...found_values) * range : Math.max(...found_values) / range;
                        }
                    }
                    perturbation_id_min.disabled = false
                    perturbation_id_max.disabled = false
                }
                else{
                    perturbation_id_status.innerText = "❌"
                    perturbation_id_min.disabled = true
                    perturbation_id_min.value = ""
                    perturbation_id_max.disabled = true
                    perturbation_id_max.value = ""
                }
            })
            perturbation_id_input.addEventListener("keydown", (e) => {
                if(e.key === "Enter"){
                    const found_values = this.get_values_from_path(perturbation_id_input.value, this.l2f.parameters)
                    if(found_values !== undefined && (typeof found_values[0] === "number")){
                        if(perturbation_groups.querySelectorAll(":scope > div").length == 0){
                            const new_perturbation_group = perturbation_group_template.content.cloneNode(true)
                            perturbation_groups.appendChild(new_perturbation_group)
                            const current_perturbation_group = perturbation_groups.querySelector(":scope > div:last-child")
                            const perturbation_slider = current_perturbation_group.querySelector(".perturbation-slider")
                            perturbation_slider.addEventListener("input", (e) => {
                                const current_perturbation_group_list = current_perturbation_group.querySelector(":scope > span.perturbation-group-list > ul")
                                const elements = Array.from(current_perturbation_group_list.querySelectorAll(":scope > li"))
                                const update_instructions = Object.fromEntries(elements.map((el, i) => {
                                    // todo: batch this
                                    const percent = parseFloat(perturbation_slider.value)
                                    const original_value = found_values[0]
                                    // const new_value = percent > 0.5 ? (percent - 0.5) * 2 * (el.max - original_value) + original_value : (1 - (0.5 - percent) * 2) * (original_value - el.min) + el.min
                                    const new_value = percent * (el.max - el.min) + el.min
                                    let new_values = l2f.states.map((state, i) => new_value)
                                    try{
                                        new_values = l2f.states.map((state, i) => eval(el.transform)(i, original_value, percent, new_value));
                                    }
                                    catch(e){
                                        console.error(e)
                                    }
                                    console.log("new value: ", new_values)

                                    el.querySelector(".perturbation-group-item-value").textContent = `${mean(new_values).toExponential(1)} ± ${std(new_values).toExponential(1)}`;
                                    return [el.path, new_values]
                                }))
                                if(elements.length > 1){
                                    const perc_value = (perturbation_slider.value * 2 - 1) * 100
                                    perturbation_slider_label.innerText = `${perc_value > 0 ? "+" : ""}${perc_value.toFixed(0)}%`
                                }
                                else{
                                    const new_values = perturbation_slider_label.innerText = update_instructions[Object.keys(update_instructions)[0]]
                                    if(new_values.every(x => x == new_values[0])){
                                        perturbation_slider_label.innerText = `${new_values[0].toExponential(2)}`
                                    }
                                    else{
                                        perturbation_slider_label.innerText = `${mean(new_values).toExponential(1)} ± ${std(new_values).toExponential(1)}`;
                                    }
                                }
                                for(const [path, new_values] of Object.entries(update_instructions)){
                                    this.set_values_at_path(path, new_values)
                                }
                            })
                            
                            const reset_button = perturbation_groups.querySelector(":scope > div:last-child button.perturbation-group-reset-button")
                            reset_button.addEventListener("click", () => {
                                const current_perturbation_group_list = current_perturbation_group.querySelector(":scope > span.perturbation-group-list > ul")
                                const elements = Array.from(current_perturbation_group_list.querySelectorAll(":scope > li"))
                                // elements.forEach((el, i) => {
                                //     perturbation_slider.value = 0.5
                                //     perturbation_slider.dispatchEvent(new Event("input"))
                                // })
                                const update_instructions = Object.fromEntries(elements.map((el, i) => {
                                    el.querySelector(".perturbation-group-item-value").textContent = "reset";
                                    return [el.path, this.get_values_from_path(el.path, this.l2f.parameters)]
                                }))
                                for(const [path, new_values] of Object.entries(update_instructions)){
                                    this.set_values_at_path(path, new_values)
                                }
                            })
                        }
                        const perturbation_group_nodes = perturbation_groups.querySelectorAll(":scope > div")
                        const current_perturbation_group = perturbation_group_nodes[perturbation_group_nodes.length - 1]
                        const perturbation_slider = current_perturbation_group.querySelector(":scope input.perturbation-slider")
                        const perturbation_slider_label = current_perturbation_group.querySelector(":scope span.control-container-label")
                        const current_perturbation_group_list = current_perturbation_group.querySelector(":scope > span.perturbation-group-list > ul")
                        const elements = Array.from(current_perturbation_group_list.querySelectorAll(":scope > li"))
                        const paths = elements.map(el => el.path)
                        let el = null
                        if(paths.includes(perturbation_id_input.value)){
                            el = elements.find(el => el.path === perturbation_id_input.value)
                            console.log("updating to: ", perturbation_id_min.value, perturbation_id_max.value)
                        }
                        else{
                            el = document.createElement("li")
                            const template = document.getElementById("perturbation-group-item-template");
                            el = template.content.firstElementChild.cloneNode(true);

                            el.path = perturbation_id_input.value
                            current_perturbation_group_list.appendChild(el)
                            el.style.cursor = "pointer"
                            el.addEventListener("click", () => {
                                current_perturbation_group_list.removeChild(el)
                                this.set_values_at_path(el.path, this.get_values_from_path(el.path, this.l2f.parameters))
                                if(current_perturbation_group_list.children.length === 0){
                                    perturbation_groups.removeChild(current_perturbation_group)
                                }
                            })
                            perturbation_slider.disabled = false;
                            perturbation_slider.min = 0
                            perturbation_slider.max = 1
                            perturbation_slider.step = 0.05
                            perturbation_slider.value = 0.5
                        }
                        el.min = parseFloat(perturbation_id_min.value)
                        el.max = parseFloat(perturbation_id_max.value)
                        el.transform = perturbation_transform.value
                        el.title = `Transform: ${el.transform}`
                        el.querySelector(".perturbation-group-item-path").textContent = el.path;
                        el.querySelector(".perturbation-group-item-range").textContent = `[${el.min.toExponential(2)}, ${el.max.toExponential(2)}]`;
                        el.querySelector(".perturbation-group-item-value").textContent = found_values.reduce((acc, c) => acc + c/found_values.length, 0).toExponential(2);
                    }
                }
            })
            perturbation_id_min.addEventListener("keydown", (e) => {
                if(e.key === "Enter"){
                    const event = new Event('keydown')
                    event.key = "Enter"
                    perturbation_id_input.dispatchEvent(event)
                }
            })
            perturbation_id_max.addEventListener("keydown", (e) => {
                if(e.key === "Enter"){
                    const event = new Event('keydown')
                    event.key = "Enter"
                    perturbation_id_input.dispatchEvent(event)
                }
            })
            perturbation_transform.addEventListener("keydown", (e) => {
                if(e.key === "Enter"){
                    const event = new Event('keydown')
                    event.key = "Enter"
                    perturbation_id_input.dispatchEvent(event)
                }
            })

            this.initialized_resolve()
        })
    }
    get_values_from_path(path, parameters){
        const objs = this.l2f.states.map((state, i) => {
            return {
                "state": JSON.parse(state.get_state()),
                "parameters": parameters[i]
            }
        })
        return path.split('.').reduce((acc, key) => key === "" ? acc : (acc && acc[0][key] !== undefined ? acc.map(x => x[key]) : undefined), objs);
    }
    set_values_at_path(instructions, new_value){
        const objs = this.l2f.states.map((state, i) => {
            return {
                "state": JSON.parse(state.get_state()),
                "parameters": structuredClone(this.l2f.perturbed_parameters[i])
            }
        })
        if(typeof instructions === "string"){
            const path = instructions
            instructions = {}
            instructions[path] = new_value
        }
        const additional_instructions = {}
        for(const [path_string, value] of Object.entries(instructions)){
            const check_path = "parameters.dynamics.J"
            if(path_string.startsWith(check_path)){
                const values = Array.isArray(value) ? value.map(x => 1/x) : 1/value
                additional_instructions["parameters.dynamics.J_inv" + path_string.slice(check_path.length)] = values
            }
        }
        const combined_instructions = {...instructions, ...additional_instructions}
        // Object.entries(additional_instructions).forEach(([path_string, value]) => {
        //     instructions[path_string] = value
        // })
        for(const [path_string, value] of Object.entries(combined_instructions)){
            const path = path_string.split('.')
            const last_key = path.pop()
            const target_objs = path.reduce((acc, key) => acc.map(x => x[key]), objs)
            target_objs.forEach((obj, i) => {
                if(Array.isArray(value)){
                    obj[last_key] = value[i]
                }
                else{
                    obj[last_key] = value 
                }
            })
        }
        this.l2f.states.forEach((state, i) => state.set_state(JSON.stringify(objs[i].state)))
        this.l2f.set_perturbed_parameters(objs.map((obj, i) => i), objs.map(obj => obj.parameters))
    }
}
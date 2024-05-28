import {make_chart} from "./ZooLearningCurves.js"

function group_by(array, keys){
    const groups = {}
    for(const item of array){
        const master_key = keys.map(key => item.config[key]).join("_")
        if(!(master_key in groups)){
            groups[master_key] = {
                keys: Object.fromEntries(keys.map(key => [key, item[key]])),
                items: []
            }
        }
        groups[master_key].items.push(item)
    }
    return groups
}


function aggregate(evaluation_data){
    const steps = []
    const returns_mean = []
    const returns_std = []
    let first_run = true
    for(const run of evaluation_data){
        if(first_run){
            steps.push(...run.data.map(entry => entry.step))
            first_run = false
            returns_mean.push(...run.data.map(entry => entry.returns_mean))
            returns_std.push(...run.data.map(entry => entry.returns_mean * entry.returns_mean))
        }
        else{
            console.assert(steps.length === run.data.length, "Steps are not the same")
            console.assert(JSON.stringify(steps) == JSON.stringify(run.data.map(entry => entry.step)), "Steps are not the same")
            for(const i in run.data){
                returns_mean[i] += run.data[i].returns_mean
                returns_std[i] += run.data[i].returns_mean * run.data[i].returns_mean
            }
        }
    }
    returns_mean.forEach((value, index) => {
        returns_mean[index] = value / evaluation_data.length
        returns_std[index] = Math.sqrt(returns_std[index] / evaluation_data.length - returns_mean[index] * returns_mean[index])
    })
    return steps.map((step, index) => {
        return {
            step: step,
            returns_mean: returns_mean[index],
            returns_std: returns_std[index]
        }
    })
}

export class Zoo{
    constructor(fs, index){
        this.container = document.createElement("div")
        this.container.classList.add("zoo-container")
        index.refresh().then(async (index) => {
            const run_list_full = index.run_list
            console.log(`Found: ${run_list_full.length} runs`)
            const run_list_zoo = run_list_full.filter((run) => run.config["name"] === "zoo")
            console.log(`Found: ${run_list_zoo.length} zoo runs`)
            const run_list = run_list_zoo.filter((run) => run.return)
            const run_list_grouped = group_by(run_list, ["population_values"])
            const run_list_grouped_truncated = Object.fromEntries(Object.entries(run_list_grouped).map(([population_values, population_runs]) => {
                const population_experiments = group_by(population_runs.items, ["experiment", "commit"])
                const experiment_keys = Object.keys(population_experiments).sort().reverse().slice(0, 3)
                return [population_values, Object.fromEntries(experiment_keys.map(key => [key, population_experiments[key]]))];
            }))
            const all_data = Object.fromEntries(await Promise.all(Object.keys(run_list_grouped_truncated).map(async population => {
                const population_data = Object.fromEntries(await Promise.all(Object.entries(run_list_grouped_truncated[population]).map(async ([experiment, experiment_runs]) => {
                    return [experiment, await Promise.all(experiment_runs.items.map(async (run) => {
                        return {
                            config: run.config,
                            label: run.config["seed"],
                            data: JSON.parse(await (await fetch(run.return)).text())
                        }
                    }))]
                })))
                return [population, population_data]
            })))


            for(const population in all_data){
                let population_actual;
                const aggregated_data = Object.entries(all_data[population]).map(([experiment, experiment_data]) => {
                    population_actual = experiment_data[0].config["population"]
                    return {
                        label: experiment,
                        data: aggregate(all_data[population][experiment])
                    }
                })
                const header = document.createElement("div")
                header.style.fontSize = "1.5em"
                header.innerHTML = `Algorithm: <b>${population_actual.algorithm}</b>, Environment: <b>${population_actual.environment}</b>`
                this.container.appendChild(header)
                const chart = make_chart(aggregated_data)
                this.container.appendChild(chart)
            }


        })
    }
    getContainer(){
        return this.container
    }

}
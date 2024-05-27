import {make_chart} from "./ZooLearningCurves.js"


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
            const all_data = await Promise.all(run_list.map(async (run) => {
                return {
                    label: run.config["experiment"],
                    data: JSON.parse(await (await fetch(run.return)).text())
                }
            }))
            const chart = make_chart(all_data)
            this.container.appendChild(chart)

        })
    }
    getContainer(){
        return this.container
    }

}
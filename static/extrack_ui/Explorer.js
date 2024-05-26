class Spoiler{
    constructor(parent, summary_text, terminal){
        this.spoiler = document.createElement('details');
        this.spoiler.classList.add("experiment-spoiler");
        this.spoiler.classList.add("spoiler");
        this.summary = document.createElement('summary');
        this.summary.classList.add("experiment-summary");
        this.summary.innerHTML = summary_text;
        this.spoiler.appendChild(this.summary);
        this.terminal = terminal;
        if(!this.terminal){
            this.child_list = document.createElement('ul');
            this.spoiler.appendChild(this.child_list)
        }
        parent.appendChild(this.spoiler);
    }
    appendChild(child){
        if(this.terminal){
            throw "Cannot append children to terminal spoiler";
        }
        this.child_list.appendChild(child);
    }
    setContent(content){
        if(!this.terminal){
            throw "Cannot set content to terminal spoiler";
        }
        this.spoiler.appendChild(content);
    }
}

export class Explorer{
    constructor(experiments_base_path){
        this.container = document.createElement('div');
        this.loading_text = document.createElement('div');
        this.loading_text.style.display = "block";
        this.container.appendChild(this.loading_text);
        const experiment_index_path = `${experiments_base_path}/index.txt`
        this.loading_text.innerHTML = `Loading Experiment Index: ${experiment_index_path}`
        const index = fetch(experiment_index_path).then(response => response.text()).then(index => {
            this.experiments = this.parseIndex(index);
            this.loading_text.style.display = "none";
            const experiment_list = document.createElement('ul');
            experiment_list.classList.add("experiment-list");
            this.container.appendChild(experiment_list);
            for (const experiment of Object.keys(this.experiments).sort().reverse()){
                const experiment_spoiler = new Spoiler(experiment_list, experiment, false);
                for (const population of Object.keys(this.experiments[experiment]).sort()) {
                    const population_spoiler = new Spoiler(experiment_spoiler, population, false);
                    for (const config of Object.keys(this.experiments[experiment][population]).sort()) {
                        const config_spoiler = new Spoiler(population_spoiler, config, false);
                        for (const seed of Object.keys(this.experiments[experiment][population][config]).sort()) {
                            const seed_spoiler = new Spoiler(config_spoiler, seed, false);
                            for (const step of Object.keys(this.experiments[experiment][population][config][seed]).sort()) {
                                const step_spoiler = new Spoiler(seed_spoiler, step, true);
                                const step_data = this.experiments[experiment][population][config][seed][step];
                                const step_info = document.createElement('div');
                                step_info.classList.add("step-info");
                                step_info.innerHTML = JSON.stringify(step_data);
                                step_spoiler.setContent(step_info);
                            }
                        }
                    }
                }
            }
        })
    }
    getContainer(){
        return this.container;
    }
    parseIndex(index){
        const regex_experiment = /^\.\/([^\/]+)/
        const regex_commit_hash = new RegExp(regex_experiment.source + /\/([^_]+)/.source)
        const regex_config_population = new RegExp(regex_commit_hash.source + /_([^\/]+)/.source)
        const regex_config = new RegExp(regex_config_population.source + /\/([^\/]+)/.source)
        const regex_seed = new RegExp(regex_config.source + /\/(\d+)/.source)
        const regex_ui_js = new RegExp(regex_seed.source + /\/ui\.js/.source)
        const regex_ui_jsm = new RegExp(regex_seed.source + /\/ui\.esm\.js/.source)
        const regex_step = new RegExp(regex_seed.source + /\/steps\/(\d+)/.source)
        const lines = index.split("\n");
        const experiments_list = [];
        for (const line of lines){
            const experiment = {};
            const experiment_match = line.match(regex_experiment);
            if (experiment_match){
                if(experiment_match[1] !== "index.txt"){
                    experiment["experiment"] = experiment_match[1];
                }
            }
            else{
                continue;
            }
            const commit_hash_match = line.match(regex_commit_hash);
            if (commit_hash_match){
                experiment["commit_hash"] = commit_hash_match[2];
            }
            else{
                continue;
            }
            const config_population_match = line.match(regex_config_population);
            if (config_population_match){
                experiment["config_population"] = config_population_match[3];
            }
            else{
                continue;
            }
            const config_match = line.match(regex_config);
            if (config_match){
                experiment["config"] = config_match[4];
            }
            else{
                continue;
            }
            const seed_match = line.match(regex_seed);
            if (seed_match){
                experiment["seed"] = seed_match[5];
            }
            else{
                continue;
            }
            if (line.match(regex_ui_js)){
                experiment["ui_js"] = true;
            }
            if (line.match(regex_ui_jsm)){
                experiment["ui_jsm"] = true;
            }
            const step_match = line.match(regex_step);
            if (step_match){
                experiment["step"] = step_match[6];
            }
            experiments_list.push(experiment);
        }
        const experiments = {}
        for(const experiment of experiments_list){
            if (experiment.experiment in experiments){
                experiments[experiment.experiment].push(experiment);
            } else {
                experiments[experiment.experiment] = [experiment];
            }
        }
        const experiments_population = {}
        for(const experiment in experiments){
            experiments_population[experiment] = {}
            for(const config of experiments[experiment]){
                const population = config.commit_hash + "_" + config.config_population;
                if (population in experiments[experiment]){
                    experiments_population[experiment][population].push(config);
                } else {
                    experiments_population[experiment][population] = [config];
                }
            }
        }
        const experiments_config = {}
        for(const experiment in experiments_population){
            experiments_config[experiment] = {}
            for(const population in experiments_population[experiment]){
                experiments_config[experiment][population] = {}
                for(const config of experiments_population[experiment][population]){
                    if (config.config in experiments_config[experiment][population]){
                        experiments_config[experiment][population][config.config].push(config);
                    } else {
                        experiments_config[experiment][population][config.config] = [config];
                    }
                }
            }
        }
        const experiments_seed = {}
        for(const experiment in experiments_config){
            experiments_seed[experiment] = {}
            for(const population in experiments_config[experiment]){
                experiments_seed[experiment][population] = {}
                for(const config in experiments_config[experiment][population]){
                    experiments_seed[experiment][population][config] = {}
                    for(const seed of experiments_config[experiment][population][config]){
                        if (seed.seed in experiments_seed[experiment][population][config]){
                            experiments_seed[experiment][population][config][seed.seed].push(seed);
                        } else {
                            experiments_seed[experiment][population][config][seed.seed] = [seed];
                        }
                    }
                }
            }
        }
        return experiments_seed;
    }
}
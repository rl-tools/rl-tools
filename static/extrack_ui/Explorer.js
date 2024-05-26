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
            throw "Cannot set content to non terminal spoiler";
        }
        this.spoiler.appendChild(content);
    }
}

export class Step{
    constructor(parent, experiments_base_path, step_paths, run){

        this.config = null
        for(const step_path of step_paths){
            if(this.config === null){
                this.config = step_path;
                this.step = this.config.step
            }
            if(step_path.checkpoint_code){
                this.checkpoint_code = step_path.path
            }
            if(step_path.checkpoint_hdf5){
                this.checkpoint_hdf5 = step_path.path
            }
            if(step_path.trajectories){
                this.trajectories = step_path.path
            }
            if(step_path.trajectories_compressed){
                this.trajectories_compressed = step_path.path
            }
        }

        const step = new Spoiler(parent, this.step, true);
        const content = document.createElement('div');
        const link = document.createElement('a');
        const url = new URL("./play_trajectories.html", window.location.href)
        url.searchParams.append("experiments", experiments_base_path)
        url.searchParams.append("trajectories", this.trajectories_compressed)
        if(!run.ui_jsm){
            throw "No ui_jsm found"
        }
        url.searchParams.append("ui", run.ui_jsm)
        link.href = url;
        link.innerText = JSON.stringify(this.config, null, 4)
        content.appendChild(link);
        step.setContent(content)
    }
}
export class Run{
    constructor(parent, experiments_base_path, experiments){
        this.config = null
        this.container = document.createElement('div');
        parent.setContent(this.container);
        for(const experiment of experiments){
            if(this.config === null){
                this.config = experiment;
                this.path = experiment.path;
            }
            if(experiment.ui_js){
                this.ui_js = experiment.path
            }
            if(experiment.ui_jsm){
                this.ui_jsm = experiment.path
            }
        }
        // second round
        this.steps_spoiler = new Spoiler(this.container, "Steps", false);
        this.steps = {}
        for(const experiment of experiments){
            if(experiment.step){
                if(experiment.step in this.steps){
                    this.steps[experiment.step].push(experiment);
                }
                else{
                    this.steps[experiment.step] = [experiment];
                }
            }
        }
        // third round
        for(const step_id in this.steps) {
            // const step_spoiler = new Spoiler(this.steps_spoiler, this.steps[step_id], );
            const step = new Step(this.steps_spoiler, experiments_base_path, this.steps[step_id], this);
        }

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
                            const seed_spoiler = new Spoiler(config_spoiler, seed, true);
                            const run = new Run(seed_spoiler, experiments_base_path, this.experiments[experiment][population][config][seed]);
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
        const regex_checkpoint_code = new RegExp(regex_step.source + /\/checkpoint\.h/.source)
        const regex_checkpoint_hdf5 = new RegExp(regex_step.source + /\/checkpoint\.h5/.source)
        const trajectories = new RegExp(regex_step.source + /\/trajectories\.json/.source)
        const trajectories_compressed = new RegExp(regex_step.source + /\/trajectories\.json.gz/.source)
        const lines = index.split("\n");
        const experiments_list = [];
        for (const line of lines){
            const experiment = {
                "path": line
            };
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
            if (line.match(regex_checkpoint_code)){
                experiment["checkpoint_code"] = true;
            }
            if (line.match(regex_checkpoint_hdf5)){
                experiment["checkpoint_hdf5"] = true;
            }
            if (line.match(trajectories)){
                experiment["trajectories"] = true;
            }
            if (line.match(trajectories_compressed)) {
                experiment["trajectories_compressed"] = true;
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
                if (population in experiments_population[experiment]){
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
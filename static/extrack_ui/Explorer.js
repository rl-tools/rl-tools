import {parseIndex} from "./ParseIndex.js";
import {TrajectoryPlayer} from "./TrajectoryPlayer.js";

class Spoiler{
    constructor(parent, summary_text, terminal, on_open, on_close){
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
        this.spoiler.addEventListener('toggle', () => {
            if(this.spoiler.open){
                if(on_open){
                    on_open();
                }
            }
            else{
                if(on_close){
                    on_close();
                }
            }
        })
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

        this.content = document.createElement('div');
        const link = document.createElement('a');
        const url = new URL("./play_trajectories.html", window.location.href)
        url.searchParams.append("experiments", experiments_base_path)
        url.searchParams.append("trajectories", this.trajectories_compressed)
        if(!run.ui_jsm){
            throw "No ui_jsm found"
        }
        url.searchParams.append("ui", run.ui_jsm)
        link.href = url.href;
        link.innerText = "Play isolated"
        this.content.appendChild(link);

        const play_button = document.createElement('button');
        play_button.innerHTML = "Play Trajectories";
        this.content.appendChild(play_button);
        this.trajectory_player_container = document.createElement('div');
        this.trajectory_player_container.classList.add("explorer-trajectory-player-container")
        this.trajectory_player_container.style.display = "none";
        this.content.appendChild(this.trajectory_player_container);
        play_button.addEventListener('click', () => {
            const trajectory_player = new TrajectoryPlayer(experiments_base_path + "/" + run.ui_jsm);
            this.trajectory_player_container.appendChild(trajectory_player.getCanvas());
            this.trajectory_player_container.style.display = "block";
            trajectory_player.playTrajectories(experiments_base_path + "/" + this.trajectories_compressed);
        })

        const step = new Spoiler(parent, this.step, true, () => {
        }, () => {
            this.trajectory_player_container.innerHTML = "";
            this.trajectory_player_container.style.display = "none";
        });
        step.setContent(this.content)
    }
}
export class Run{
    constructor(parent, experiments_base_path, experiments){
        this.config = null
        this.container = document.createElement('div');
        this.container.classList.add("run-container");
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
            this.experiments = parseIndex(index);
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
}
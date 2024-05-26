import * as pako from 'https://cdn.jsdelivr.net/npm/pako@2.0.4/+esm';

async function fetchAndDecompressData(url) {
    const response = await fetch(url);
    const compressedData = await response.arrayBuffer();
    const decompressedData = pako.ungzip(new Uint8Array(compressedData), { to: 'string' });
    return JSON.parse(decompressedData);
}

export class TrajectoryPlayer{
    constructor(render_function_path) {
        // const experiments_stub = "../../experiments";
        // const modulePath = `${experiments_stub}/experiments/2024-05-25_14-28-34/32e6580_zoo_algorithm_environment/sac_pendulum-v1/0000/ui.esm.js`;

        this.render = import(render_function_path)
            .then(({ render }) => {
                return render
            })
        this.container = document.createElement('div');
        this.container.classList.add("trajectory-player-container")
        this.canvas_container = document.createElement('div');
        this.canvas_container.classList.add("trajectory-player-canvas-container")
        this.canvas = document.createElement('canvas');
        this.canvas.classList.add("trajectory-player-canvas")
        this.canvas.style.display = "none";
        this.canvas_container.appendChild(this.canvas);
        this.container.appendChild(this.canvas_container);
        this.loading_text = document.createElement('p');
        this.loading_text.style.display = "none";
        this.container.appendChild(this.loading_text);
        this.controls_container = document.createElement('div');
        this.controls_container.classList.add("trajectory-player-controls-container");

        this.container.appendChild(this.controls_container);
    }
    getCanvas(){
        return this.container;
    }
    async playTrajectories(path) {
        this.loading_text.innerHTML = `Loading Trajectory Data from ${path}`
        this.loading_text.style.display = "inline";
        const trajectoryData = await fetchAndDecompressData(path);
        this.loading_text.style.display = "none";
        this.canvas.style.display = "inline"
        let currentEpisode = 0;
        let currentStep = 0;
        let currentEpisodeLength = 0;
        let currentEpisodeReturn = 0;

        const render = await this.render;

        const episode_info = document.createElement('div');
        episode_info.classList.add("trajectory-player-episode-info");

        const skip_button = document.createElement('button');
        skip_button.innerHTML = "Skip to Next Episode";
        this.controls_container.replaceChildren(episode_info, skip_button);


        const step = () => {
            const size = Math.min(this.canvas_container.clientWidth, this.canvas_container.clientHeight);
            this.canvas.width = size;
            this.canvas.height = size;
            const ctx = this.canvas.getContext('2d');
            episode_info.innerHTML = `Path: ${path}`;
            episode_info.innerHTML += `</br>Episode: ${currentEpisode}, Step: ${currentStep}, Return: ${currentEpisodeReturn.toFixed(2)}`;
            if (currentEpisode < trajectoryData.length) {
                const episode = trajectoryData[currentEpisode];
                if (currentStep < episode.length) {
                    const { state, action, reward, terminated } = episode[currentStep];
                    render(ctx, state, action);
                    currentEpisodeLength++;
                    currentEpisodeReturn += reward;
                    currentStep++;
                } else {
                    currentStep = 0;
                    currentEpisode++;
                    if(currentEpisode >= trajectoryData.length){
                        currentEpisode = 0;
                    }
                    console.log(`Episode ${currentEpisode} finished. Return = ${currentEpisodeReturn}, Length = ${currentEpisodeLength}`);
                    currentEpisodeLength = 0;
                    currentEpisodeReturn = 0;
                }

            }
        }

        function skipEpisode() {
            currentStep = 0;
            currentEpisode++;
            if (currentEpisode >= trajectoryData.length) {
                currentEpisode = 0;
            }
            console.log(`Skipped to Episode ${currentEpisode}`);
        }

        skip_button.addEventListener('click', skipEpisode);

        setInterval(step, 50);
    }

}


import {Chart, LineController, ScatterController, LineElement, PointElement, LinearScale, CategoryScale, Title, Tooltip, Filler, Legend} from 'https://cdn.skypack.dev/chart.js@4';
Chart.register(LineController, ScatterController, LineElement, PointElement, LinearScale, CategoryScale, Title, Tooltip, Filler, Legend);

function make_chart(data) {
    const container = document.createElement("div");
    container.classList.add("zoo-chart-container");
    const canvas = document.createElement("canvas");
    canvas.classList.add("zoo-chart");
    container.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    // Calculate the upper and lower bounds of the standard deviation range
    const upperBound = data.map(step => step["returns_mean"] + step["returns_std"]);
    const lowerBound = data.map(step => step["returns_mean"] - step["returns_std"]);

    const myChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(step => step["step"]),
            datasets: [
                {
                    label: 'Returns Mean',
                    data: data.map(step => step["returns_mean"]),
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    yAxisID: 'y',
                    fill: false // Ensure the mean line itself is not filled
                },
                {
                    label: '+Standard Deviation',
                    data: upperBound,
                    borderColor: 'rgba(75, 192, 192, 0)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    yAxisID: 'y',
                    fill: '+1', // Fill from this dataset to the next one
                    pointRadius: 0,
                    pointHoverRadius: 0,
                },
                {
                    label: '-Standard Deviation',
                    data: lowerBound,
                    borderColor: 'rgba(75, 192, 192, 0)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    yAxisID: 'y',
                    fill: false, // Ensure the lower bound line is not filled
                    pointRadius: 0,
                    pointHoverRadius: 0,
                }
            ]
        },
        options: {
            scales: {
                y: {
                    type: 'linear',
                    position: 'left',
                    ticks: {
                        beginAtZero: true
                    },
                    title: {
                        display: true,
                        text: 'Return'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Step'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        filter: function(legendItem, chartData) {
                            return legendItem.text !== '+Standard Deviation' && legendItem.text !== '-Standard Deviation';
                        }
                    }
                }
            }
        }
    });
    return container;
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
            for(const run of run_list){
                const run_container = document.createElement("div")
                run_container.classList.add("run-container")
                run_container.innerText = run.return
                const data = JSON.parse(await (await fetch(run.return)).text())
                const chart = make_chart(data)
                run_container.appendChild(chart)
                this.container.appendChild(run_container)

            }

        })
    }
    getContainer(){
        return this.container
    }

}
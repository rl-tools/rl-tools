import {Chart, LineController, ScatterController, LineElement, PointElement, LinearScale, CategoryScale, Title, Tooltip, Filler, Legend} from 'https://cdn.skypack.dev/chart.js@4';
Chart.register(LineController, ScatterController, LineElement, PointElement, LinearScale, CategoryScale, Title, Tooltip, Filler, Legend);

function hexToRgba(hex, alpha = 1.0) {
    // Remove the hash at the start if it's there
    hex = hex.replace(/^#/, '');

    // Parse r, g, b values
    let r = parseInt(hex.substring(0, 2), 16);
    let g = parseInt(hex.substring(2, 4), 16);
    let b = parseInt(hex.substring(4, 6), 16);

    // Return the RGBA string
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// Example usage
const hexColor = "#e6f598";
const rgbaColor = hexToRgba(hexColor, 0.8); // Second parameter is the alpha value (opacity)
console.log(rgbaColor); // Output: "rgba(230, 245, 152, 0.8)"

function make_chart(data) {
    const container = document.createElement("div");
    container.classList.add("zoo-chart-container");
    const canvas = document.createElement("canvas");
    canvas.classList.add("zoo-chart");
    container.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    const steps = Array.from(new Set(data.reduce((a, c) => [...a, ...c.data.map(step => step.step)], []))).sort((a, b) => a - b);
    for (const d of data) {
        d.mean_returns = steps.map(step => {
            const entry = d.data.find(d => d.step === step);
            return entry ? entry.returns_mean : null;
        });
        d.std_returns = steps.map(step => {
            const entry = d.data.find(d => d.step === step);
            return entry ? entry.returns_std : null;
        });
        d.mean_plus_std = d.mean_returns.map((mean, index) => {
            const std = d.std_returns[index];
            return mean !== null && std !== null ? mean + std : null;
        });

        d.mean_minus_std = d.mean_returns.map((mean, index) => {
            const std = d.std_returns[index];
            return mean !== null && std !== null ? mean - std : null;
        });
    }

    let color_palette = [
        "#d53e4f",
        "#fc8d59",
        "#fee08b",
        "#ffffbf",
        "#e6f598",
    ]

    let datasets = [];
    for (const d of data) {
        const color_hex = color_palette.shift()
        const color = hexToRgba(color_hex);
        const color_ribbon = hexToRgba(color_hex, 0.2);
        datasets.push({
            label: d.label,
            data: d.mean_returns,
            borderColor: color,
            backgroundColor: color_ribbon,
            yAxisID: 'y',
            fill: false,
            spanGaps: true
        });
        datasets.push({
            label: `${d.label} +Standard Deviation`,
            data: d.mean_plus_std,
            borderColor: 'rgba(0, 0, 0, 0)',
            backgroundColor: color_ribbon,
            yAxisID: 'y',
            fill: '+1',
            pointRadius: 0,
            pointHoverRadius: 0,
            spanGaps: true
        });
        datasets.push({
            label: `${d.label} -Standard Deviation`,
            data: d.mean_minus_std,
            borderColor: 'rgba(0, 0, 0, 0)',
            backgroundColor: color_ribbon,
            yAxisID: 'y',
            fill: false,
            pointRadius: 0,
            pointHoverRadius: 0,
            spanGaps: true
        });
    }


    const myChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: steps,
            datasets: datasets
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
                            return !legendItem.text.endsWith('Standard Deviation');
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
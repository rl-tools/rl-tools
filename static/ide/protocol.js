// The training program's stdout is the interface. These are the lines the RLtools loop steps print today (evaluation and timing steps).
const NUMBER = "(-?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][-+]?\\d+)?|-?nan|-?inf)";
const EVALUATION = new RegExp(`^Step: (\\d+)/(\\d+) Mean return: ${NUMBER} Mean episode length: ${NUMBER}`);
const STEPS_PER_SECOND = new RegExp(`^Loop step: (\\d+), env step: (\\d+), SPS: ${NUMBER}`);
const TOTAL_TIME = new RegExp(`^Time: ${NUMBER}s`);

export function parseLine(line){
    let match = EVALUATION.exec(line);
    if(match){
        return { type: "evaluation", step: Number(match[1]), stepLimit: Number(match[2]), meanReturn: Number(match[3]), meanEpisodeLength: Number(match[4]) };
    }
    match = STEPS_PER_SECOND.exec(line);
    if(match){
        return { type: "steps_per_second", step: Number(match[1]), environmentStep: Number(match[2]), stepsPerSecond: Number(match[3]) };
    }
    match = TOTAL_TIME.exec(line);
    if(match){
        return { type: "total_time", seconds: Number(match[1]) };
    }
    return null;
}

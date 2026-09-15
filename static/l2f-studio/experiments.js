const CHECKPOINT_PATTERN = /^\.\/([^\/]+)\/([^\/]+)\/([^\/]+)\/(\d+)\/steps\/(\d+)\/(checkpoint[^\/]*\.h5)$/

function encodePath(path) {
    return path.split("/").map(encodeURIComponent).join("/")
}

export function parseExperimentCheckpoints(index, base_path) {
    const checkpoints = []
    for (const line of index.split("\n")) {
        const match = line.trim().match(CHECKPOINT_PATTERN)
        if (!match) {
            continue
        }
        const [, experiment, population, config, seed, step, file] = match
        const run = `${experiment}/${population}/${config}/${seed}`
        const path = `${run}/steps/${step}/${file}`
        checkpoints.push({ experiment, population, config, seed, step, file, run, path, url: base_path + encodePath(path) })
    }
    checkpoints.sort((a, b) => {
        if (a.experiment !== b.experiment) {
            return b.experiment.localeCompare(a.experiment)
        }
        if (a.run !== b.run) {
            return a.run.localeCompare(b.run)
        }
        if (a.step !== b.step) {
            return Number(b.step) - Number(a.step)
        }
        return a.file.localeCompare(b.file)
    })
    return checkpoints
}

export async function fetchExperimentCheckpoints(base_path) {
    const response = await fetch(`${base_path}index.txt`, { cache: "no-store" })
    if (!response.ok) {
        return null
    }
    return parseExperimentCheckpoints(await response.text(), base_path)
}

export function filterExperimentCheckpoints(checkpoints, filter) {
    const tokens = filter.toLowerCase().split(/\s+/).filter(token => token.length > 0)
    if (tokens.length === 0) {
        return checkpoints
    }
    return checkpoints.filter(checkpoint => {
        const haystack = checkpoint.path.toLowerCase()
        return tokens.every(token => haystack.includes(token))
    })
}

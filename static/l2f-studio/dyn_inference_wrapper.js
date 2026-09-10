import createModule from "dyn-inference"

let modulePromise = null

function getModule() {
    if (!modulePromise) modulePromise = createModule()
    return modulePromise
}

export async function load(input) {
    let arrayBuffer
    if (typeof input === "string") {
        arrayBuffer = await (await fetch(input)).arrayBuffer()
    } else if (input instanceof ArrayBuffer) {
        arrayBuffer = input
    } else {
        throw new Error("Input must be a URL string or ArrayBuffer")
    }

    const module = await getModule()
    const path = "/model_" + Date.now() + ".h5"
    module.FS.writeFile(path, new Uint8Array(arrayBuffer))

    const inference = new module.DynInference()
    if (!inference.load(path)) {
        module.FS.unlink(path)
        throw new Error("Failed to load model from HDF5")
    }
    module.FS.unlink(path)

    let meta = null
    try { meta = JSON.parse(inference.get_meta()) } catch (e) { meta = {} }

    const num_branches = inference.get_num_branches()
    const input_dims = inference.get_input_dims()

    return {
        checkpoint_name: inference.get_checkpoint_name(),
        meta,
        input_dim: inference.get_input_dim(),
        output_dim: inference.get_output_dim(),
        num_branches,
        input_dims,
        create_state() { return inference.create_state() },
        reset_state(id) { inference.reset_state(id) },
        evaluate_step(input_array, state_id) {
            return Array.from(inference.evaluate_step(state_id, input_array))
        },
        evaluate(input_array) {
            return Array.from(inference.evaluate(input_array))
        },
        evaluate_tuple(branch_arrays) {
            return Array.from(inference.evaluate_tuple(branch_arrays))
        },
        verify() {
            return inference.verify()
        },
        destroy() { inference.destroy() },
        description() {
            return `${inference.get_checkpoint_name()} (${inference.get_input_dim()} → ${inference.get_output_dim()})`
        },
    }
}

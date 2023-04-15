import createBackpropToolsInterfaceBenchmark from './build/wasm_interface_benchmark.js';
import createBackpropToolsInterface from './build/wasm_interface.js';

let mode = null;
let bpt = null;
let training_state = null;
let training_finished = false;
self.addEventListener("message", async (event) => {
    console.log("Message received from main script");
    if (event.data.type === 'initialize') {
        console.log("Initializing worker, benchmark: " + event.data.payload.benchmark);
        console.assert(bpt === null)

        mode = event.data.payload.benchmark ? 'benchmark' : 'normal';
        const bpt_interface_factory = mode === "benchmark" ? createBackpropToolsInterfaceBenchmark : createBackpropToolsInterface;

        bpt = await bpt_interface_factory().then((bpt_emscripten) => {
            console.log("Initializing worker");
            return {
                create_training_state: bpt_emscripten._proxy_create_training_state,
                training_step: bpt_emscripten._proxy_training_step,
                destroy_training_state: bpt_emscripten._proxy_destroy_training_state,
                get_step: bpt_emscripten._proxy_get_step,
                get_evaluation_count: bpt_emscripten._proxy_get_evaluation_count,
                get_evaluation_return: bpt_emscripten._proxy_get_evaluation_return
            }
        })

        self.postMessage({id: event.data.id, type: 'worker_initialization_finished'});
        return
    }
    if (event.data.type === 'initialize_training_state') {
        console.assert(bpt !== null)
        training_state = bpt.create_training_state();
        self.postMessage({id: event.data.id, type: 'finished_initializing_training_state'});
        return
    }
    if (event.data.type === 'full_training') {
        console.assert(training_state !== null)
        while(!training_finished) {
            let step = bpt.get_step(training_state);
            if (step % 100 === 0) {
                self.postMessage({type: 'training_step', payload: {step: step}});
            }
            training_finished = bpt.training_step(training_state);
        }
        if(mode === 'benchmark') {
            let evaluation_returns = []
            let num_evaluations = bpt.get_evaluation_count();
            for(let i = 0; i < num_evaluations; i++){
                let evaluation_return = bpt.get_evaluation_return(ts, i);
                evaluation_returns.push(evaluation_return);
            }
            self.postMessage({id: event.data.id, type: 'finished_training', payload: {evaluation_returns: evaluation_returns}});
        }
        else{
            self.postMessage({id: event.data.id, type: 'finished_training'});
        }
        return
    }
    if (event.data.type === 'reset_training') {
        console.assert(training_state !== null)
        bpt.destroy_training_state(training_state);
        training_state = bpt.create_training_state();
        self.postMessage({id: event.data.id, type: 'finished_reset'});
        return
    }
    if (event.data.type === 'destroy_training_state') {
        console.assert(training_state !== null)
        bpt.destroy_training_state(training_state);
        training_state = null;
        self.postMessage({id: event.data.id, type: 'finished_destroying_training_state'});
        return
    }
    console.warn("Unknown message type: " + event.data.type)
})

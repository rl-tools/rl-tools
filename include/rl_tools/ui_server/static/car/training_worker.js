import createRLtoolsInterface from './build/wasm_interface.js';

let mode = null;
let rlt = null;
let training_state = null;
let training_finished = false;
let state_dim = null;
let current_episode = null;

async function async_main(){
    const loaded = await createRLtoolsInterface().then((rlt_emscripten) => {
        console.log("Initializing worker");
        self.addEventListener("message", async (event) => {
            console.log("Message received from main script: ", event.data);
        })
        return {
            module: rlt_emscripten,
            bindings: {
                proxy_create: rlt_emscripten._proxy_create,
                proxy_step: rlt_emscripten._proxy_step,
                proxy_step_message: rlt_emscripten._proxy_step_message,
                proxy_destroy: rlt_emscripten._proxy_destroy,
                proxy_num_messages: rlt_emscripten._proxy_num_messages,
                proxy_pop_message: rlt_emscripten._proxy_pop_message,
                proxy_delete_message: rlt_emscripten._proxy_delete_message,
            }
        }
    })
    const rlt = loaded.bindings;
    let training_state = rlt.proxy_create(0);
    console.log("Training state: ", training_state);
    console.log("Num messages: ", )
    while(rlt.proxy_num_messages(training_state) > 0){
        const message_pointer = rlt.proxy_pop_message(training_state);
        const message = loaded.module.UTF8ToString(message_pointer);
        
    }
}
async_main()
    // console.log("Message received from main script: ", event.data);

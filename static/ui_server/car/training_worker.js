import createRLtoolsInterface from './build/wasm_interface.js';

let mode = null;
let rlt = null;
let training_state = null;
let training_finished = false;
let state_dim = null;
let current_episode = null;

async function async_main(){
    const rlt = await createRLtoolsInterface()
    console.log("Initializing worker");
    let training_state = rlt._proxy_create(0);
    console.log("Training state: ", training_state);

    let messages = []
    let playbackSpeed = 1;
    self.addEventListener("message", async (event) => {
        if(event.data.channel === "setPlaybackSpeed"){
            playbackSpeed = event.data.data;
            console.log("Setting playback speed to: ", playbackSpeed);
        }
        else{
            messages.push(event.data)
        }
    })


    let main = async () =>{
        let sleep = 0;
        while(messages.length > 0){
            const message = messages.shift();
            const message_string = JSON.stringify(message)
            let message_ptr = rlt.stringToNewUTF8(message_string);
            sleep += rlt._proxy_step_message(training_state, message_ptr)
        }
        for(let i = 0; i < 100; i++){
            sleep += rlt._proxy_step(training_state)
        }
        while(rlt._proxy_num_messages(training_state) > 0){
            const message_pointer = rlt._proxy_pop_message(training_state);
            const message = rlt.UTF8ToString(message_pointer);
            self.postMessage(JSON.parse(message))
            await new Promise(resolve => setTimeout(resolve, 10/playbackSpeed));
        }
        setTimeout(main, sleep);
    }
    main()
}
async_main()
    // console.log("Message received from main script: ", event.data);

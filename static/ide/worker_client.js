export class WorkerClient{
    constructor(worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" })){
        this.worker = worker;
        this.pending = null;
        this.sequence = 0;
        this.closed = false;
        worker.addEventListener("message", ({ data }) => {
            const pending = this.pending;
            if(!pending || data.id !== pending.id){
                return;
            }
            if(data.type === "event"){
                try{
                    pending.onEvent(data.event);
                }
                catch(error){
                    this.terminate(error);
                }
            }
            else if(data.type === "result"){
                this.pending = null;
                clearTimeout(pending.timer);
                pending.resolve(data.result);
            }
            else if(data.type === "error"){
                this.terminate(new Error(data.message));
            }
        });
        worker.addEventListener("error", event => this.terminate(new Error(event.message || "worker failed")));
        worker.addEventListener("messageerror", () => this.terminate(new Error("worker message could not be decoded")));
    }
    request(type, payload = {}, { timeout = 300000, transfer = [], onEvent = () => {} } = {}){
        if(this.closed || this.pending){
            return Promise.reject(new Error(this.closed ? "worker is closed" : "worker is busy"));
        }
        return new Promise((resolve, reject) => {
            const id = ++this.sequence;
            const timer = setTimeout(() => this.terminate(new Error(type + " timed out")), timeout);
            this.pending = { id, resolve, reject, timer, onEvent };
            try{
                this.worker.postMessage({ ...payload, id, type }, transfer);
            }
            catch(error){
                this.terminate(error);
            }
        });
    }
    terminate(reason = new Error("cancelled")){
        if(this.closed){
            return;
        }
        this.closed = true;
        this.worker.terminate();
        if(this.pending){
            clearTimeout(this.pending.timer);
            this.pending.reject(reason);
            this.pending = null;
        }
    }
}

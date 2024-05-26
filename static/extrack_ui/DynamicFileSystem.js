export class DynamicFileSystem{
    constructor(base_path){
        this.base_path = base_path;
    }
    async refresh(node){
        return this.loadTree(node.path)
    }
    async loadTree(relative_path){
        if(!relative_path){
            relative_path = ""
        }
        const current_base_path = `${this.base_path}${relative_path}`
        const files_promise = fetch(`${current_base_path}/index_files.txt`)
            .then(response => {
                if(!response.ok){
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.text()
            })
            .then(index => {
                return index.split("\n").filter(line => line.length > 0).map((name) =>{
                    const file = {}
                    file[name] = `${current_base_path}/${name}`
                    return file
                })
            })
        const directories_promise = fetch(`${current_base_path}/index_directories.txt`)
            .then(response => {
                if(!response.ok){
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.text()
            })
            .then(index => {
                return Promise.all(index.split("\n").filter(line => line.length > 0).map(async (name) =>{
                    const directory = {}
                    directory[name] = await this.loadTree(`${relative_path}/${name}`)
                    return directory
                }))
            })

        const files = await files_promise;
        const directories = await directories_promise;
        const children = Object.assign({}, ...files, ...directories)
        return {
            "children": children,
            "path": current_base_path
        }

    }
}
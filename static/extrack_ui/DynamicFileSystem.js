export class FileSystem{
    constructor(base_path){
        this.base_path = base_path;
    }
    async loadTree(base_path){
         const files_promise = fetch(`${base_path}/index_files.txt`)
            .then(response => {
                if(!response.ok){
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.text()
            })
            .then(index => {
                return index.split("\n").filter(line => line.length > 0)
            })
         const directories_promise = fetch(`${base_path}/index_directories.txt`)
            .then(response => {
                if(!response.ok){
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.text()
            })
            .then(index => {
                return index.split("\n").filter(line => line.length > 0)
            })
    }
    normalize(path){
        // this should remove all relative path components like test/../hello.txt => test/hello.txt
        return (new URL(path, window.location.origin)).pathname.substring(1)
    }
    addNode(node, path, relativePath, fullPath) {
        const firstSeparatorIndex = path.indexOf('/');
        let name;
        let leaf = false;
        let remainder;
        if (firstSeparatorIndex === -1) {
            name = path.substring(0)
            leaf = true
        } else {
            if (firstSeparatorIndex === 0) {
                throw new Error(`Path ${path} starts with a separator (full path: ${fullPath})`)
            }
            name = path.substring(0, firstSeparatorIndex)
            remainder = path.substring(firstSeparatorIndex+1)
        }
        relativePath = relativePath + name + "/"
        if(leaf) {
            if(name in node.children) {
                throw new Error(`Duplicate node ${name} at ${fullPath}`)
            }
            node.children[name] = fullPath
        }
        else {
            if(!(name in node.children)) {
                node.children[name] = {
                    children: {},
                    path: relativePath
                }
            }
            this.addNode(node.children[name], remainder, relativePath, fullPath)
        }

    }
    parsePath(tree, path){
        const normalized = this.normalize(path)
        this.addNode(tree, normalized, "", normalized)
        console.log(normalized)
    }
    parse(index){
        const tree = {
            children: {},
            path: ""
        }
        for(const line of index.split("\n")){
            if(line.length > 0){
                this.parsePath(tree, line)
            }
        }
        return tree
    }
    async ls(path){

    }
    async load(node, path){
    }
}
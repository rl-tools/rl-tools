import { File, Directory } from "./wasi.js";

const encoder = new TextEncoder();

function toBytes(data){
    return typeof data === "string" ? encoder.encode(data) : data;
}

function insert(root, path, data){
    const parts = path.split("/").filter(part => part.length > 0 && part !== ".");
    let node = root;
    for(const part of parts.slice(0, -1)){
        let child = node.get(part);
        if(!(child instanceof Map)){
            child = new Map();
            node.set(part, child);
        }
        node = child;
    }
    node.set(parts[parts.length - 1], data);
}

function materialize(node, readonly){
    const contents = new Map();
    for(const [name, child] of node){
        contents.set(name, child instanceof Map ? new Directory(materialize(child, readonly), { readonly }) : new File(toBytes(child), { readonly }));
    }
    return contents;
}

// Builds the contents map of a directory (name -> File | Directory) from path -> data entries
export function treeFromEntries(entries, { readonly = false } = {}){
    const root = new Map();
    for(const [path, data] of entries){
        insert(root, path, data);
    }
    return materialize(root, readonly);
}

// Merges several trees into one contents map; later trees win on name collisions
export function mergeTrees(...trees){
    const merged = new Map();
    for(const tree of trees){
        for(const [name, inode] of tree){
            merged.set(name, inode);
        }
    }
    return merged;
}

// Inverse of treeFromEntries: path -> Uint8Array for every file below the given contents map
export function entriesFromTree(tree, prefix = ""){
    const entries = new Map();
    for(const [name, inode] of tree){
        const path = prefix.length > 0 ? prefix + "/" + name : name;
        if(inode instanceof Directory){
            for(const [childPath, data] of entriesFromTree(inode.contents, path)){
                entries.set(childPath, data);
            }
        }
        else{
            entries.set(path, inode.data);
        }
    }
    return entries;
}

export function readFile(tree, path){
    let node = tree;
    const parts = path.split("/").filter(part => part.length > 0 && part !== ".");
    for(const part of parts.slice(0, -1)){
        const child = node.get(part);
        if(!(child instanceof Directory)){
            return null;
        }
        node = child.contents;
    }
    const file = node.get(parts[parts.length - 1]);
    return file instanceof File ? file.data : null;
}

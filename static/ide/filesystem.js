import { File, Directory } from "./wasi.js";
import { pathParts } from "./path.js";

const encoder = new TextEncoder();

export function treeFromEntries(entries, { readonly = false } = {}){
    const root = new Map();
    for(const [path, data] of entries){
        const parts = pathParts(path);
        if(parts.length === 0){
            throw new Error("a file must have a name");
        }
        let contents = root;
        for(const part of parts.slice(0, -1)){
            if(!contents.has(part)){
                contents.set(part, new Directory(new Map(), { readonly }));
            }
            const directory = contents.get(part);
            if(!(directory instanceof Directory)){
                throw new Error(path + ": a parent is a file");
            }
            contents = directory.contents;
        }
        const name = parts[parts.length - 1];
        if(contents.get(name) instanceof Directory){
            throw new Error(path + ": cannot replace a directory with a file");
        }
        contents.set(name, new File(typeof data === "string" ? encoder.encode(data) : data, { readonly }));
    }
    return root;
}

export function entriesFromTree(tree, prefix = "", entries = new Map()){
    for(const [name, inode] of tree){
        const path = prefix + name;
        if(inode instanceof Directory){
            entriesFromTree(inode.contents, path + "/", entries);
        }
        else{
            entries.set(path, inode.data);
        }
    }
    return entries;
}

export function readFile(tree, path){
    const parts = pathParts(path);
    let contents = tree;
    for(const part of parts.slice(0, -1)){
        const directory = contents.get(part);
        if(!(directory instanceof Directory)){
            return null;
        }
        contents = directory.contents;
    }
    const file = contents.get(parts[parts.length - 1]);
    return file instanceof File ? file.data : null;
}

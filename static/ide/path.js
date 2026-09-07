export function pathParts(path){
    if(path.includes("\0")){
        throw new Error("paths cannot contain NUL");
    }
    const parts = [];
    for(const part of path.split("/")){
        if(part === "" || part === "."){
            continue;
        }
        if(part === ".."){
            if(parts.length === 0){
                throw new Error("path leaves the filesystem root");
            }
            parts.pop();
        }
        else{
            parts.push(part);
        }
    }
    return parts;
}

export function normalizePath(path){
    return pathParts(path).join("/");
}

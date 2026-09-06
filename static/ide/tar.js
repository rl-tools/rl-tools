const decoder = new TextDecoder();

function readString(bytes, offset, length){
    let end = offset;
    const limit = offset + length;
    while(end < limit && bytes[end] !== 0){
        end++;
    }
    return decoder.decode(bytes.subarray(offset, end));
}

function readOctal(bytes, offset, length){
    const text = readString(bytes, offset, length).trim();
    return text.length === 0 ? 0 : parseInt(text, 8);
}

function normalizePath(path){
    return path.split("/").filter(part => part.length > 0 && part !== ".").join("/");
}

// Reads a ustar/GNU tar archive into a map from normalized path to file contents (directories and metadata entries are skipped)
export function untar(bytes){
    const entries = new Map();
    let offset = 0;
    let longName = null;
    while(offset + 512 <= bytes.length){
        if(bytes[offset] === 0){
            break;
        }
        const name = readString(bytes, offset, 100);
        const size = readOctal(bytes, offset + 124, 12);
        const typeflag = String.fromCharCode(bytes[offset + 156] || 48);
        const prefix = readString(bytes, offset + 345, 155);
        const dataStart = offset + 512;
        const path = longName ?? (prefix.length > 0 ? prefix + "/" + name : name);
        longName = null;
        if(typeflag === "L"){
            longName = readString(bytes, dataStart, size);
        }
        else if(typeflag === "0"){
            entries.set(normalizePath(path), bytes.subarray(dataStart, dataStart + size));
        }
        offset = dataStart + Math.ceil(size / 512) * 512;
    }
    return entries;
}

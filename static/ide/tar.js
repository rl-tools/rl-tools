import { normalizePath } from "./path.js";

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

// Both package producers use ustar; unsupported entries must not silently disappear from the mounted filesystem.
export function untar(bytes){
    if(bytes.length % 512 !== 0){
        throw new Error("truncated ustar archive");
    }
    const entries = new Map();
    let offset = 0;
    while(offset + 512 <= bytes.length){
        if(bytes[offset] === 0){
            break;
        }
        if(readString(bytes, offset + 257, 6) !== "ustar" || readString(bytes, offset + 263, 2) !== "00"){
            throw new Error("unsupported archive format: expected ustar");
        }
        const name = readString(bytes, offset, 100);
        const size = readOctal(bytes, offset + 124, 12);
        const typeflag = String.fromCharCode(bytes[offset + 156] || 48);
        const prefix = readString(bytes, offset + 345, 155);
        const dataStart = offset + 512;
        const path = prefix.length > 0 ? prefix + "/" + name : name;
        if(!Number.isSafeInteger(size) || size < 0 || dataStart + size > bytes.length){
            throw new Error(`${path}: truncated or invalid ustar entry`);
        }
        if(typeflag === "0"){
            entries.set(normalizePath(path), bytes.subarray(dataStart, dataStart + size));
        }
        else if(typeflag !== "5"){
            throw new Error(`${path}: unsupported ustar entry type ${typeflag}`);
        }
        offset = dataStart + Math.ceil(size / 512) * 512;
    }
    return entries;
}

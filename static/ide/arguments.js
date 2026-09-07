export function parseArguments(text){
    const args = [];
    let token = "";
    let quote = null;
    let present = false;
    for(let i = 0; i < text.length; i++){
        const character = text[i];
        if(character === "\\" && quote !== "'"){
            if(i + 1 === text.length){
                throw new Error("unfinished escape in arguments");
            }
            const next = text[i + 1];
            if(quote === '"' && !['"', "\\", "$"].includes(next)){
                token += character;
            }
            else{
                token += text[++i];
            }
            present = true;
        }
        else if(quote !== null){
            if(character === quote){
                quote = null;
            }
            else{
                token += character;
            }
        }
        else if(character === '"' || character === "'"){
            quote = character;
            present = true;
        }
        else if(/\s/.test(character)){
            if(present){
                args.push(token);
                token = "";
                present = false;
            }
        }
        else{
            token += character;
            present = true;
        }
    }
    if(quote !== null){
        throw new Error("unclosed quote in arguments");
    }
    if(present){
        args.push(token);
    }
    return args;
}

export function formatArguments(args){
    return args.map(argument => /^[\w./=:+,-]+$/.test(argument) ? argument : '"' + argument.replace(/["\\$]/g, character => "\\" + character) + '"').join(" ");
}

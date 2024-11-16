export class DynamicFileSystem {
    constructor(base_path) {
        this.base_path = base_path;
    }

    async loadTree() {
        return await this.fetchAndParseDirectory("");
    }

    async fetchDirectory(path) {
        const url = this.base_path + path;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.text();
    }

    parseDirectoryHtml(html) {
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const links = Array.from(doc.getElementsByTagName('a'));

        return links
            .map(link => ({
                name: link.textContent,
                isDirectory: link.textContent.endsWith('/')
            }))
            .filter(({name}) => name !== '../' && name.length > 0)
            .map(({name, isDirectory}) => ({
                name: isDirectory ? name.slice(0, -1) : name,
                isDirectory
            }));
    }

    async fetchAndParseDirectory(relativePath) {
        const html = await this.fetchDirectory(relativePath);
        const entries = this.parseDirectoryHtml(html);

        const node = {
            children: {},
            path: relativePath
        };

        // Create an array of promises for directory fetches
        const directoryPromises = entries.map(async ({name, isDirectory}) => {
            if (isDirectory) {
                const childPath = relativePath + name + "/";
                // Return both the name and the promise result
                return {
                    name,
                    children: await this.fetchAndParseDirectory(childPath),
                    isDirectory: true
                };
            } else {
                return {
                    name,
                    path: this.base_path + relativePath + name,
                    isDirectory: false
                };
            }
        });

        // Wait for all directory fetches to complete concurrently
        const results = await Promise.all(directoryPromises);

        // Populate the node's children from the results
        for (const result of results) {
            if (result.isDirectory) {
                node.children[result.name] = result.children;
            } else {
                node.children[result.name] = result.path;
            }
        }

        return node;
    }

    normalize(path) {
        return (new URL(path, window.location.origin)).pathname.substring(1);
    }

    // For compatibility with existing interface
    addNode(node, path, relativePath, fullPath) {
        throw new Error('addNode is not used in DynamicFileSystem');
    }

    parsePath(tree, path) {
        throw new Error('parsePath is not used in DynamicFileSystem');
    }

    parse(index) {
        throw new Error('parse is not used in DynamicFileSystem');
    }
}


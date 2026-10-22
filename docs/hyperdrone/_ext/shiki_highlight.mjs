import { createHighlighter } from "shiki";

let input = "";
for await (const chunk of process.stdin) {
    input += chunk;
}

const highlighter = await createHighlighter({
    themes: ["github-light", "github-dark"],
    langs: ["python"],
});
const rendered = JSON.parse(input).map(({ code, decorations }) => highlighter.codeToHtml(code, {
    lang: "python",
    themes: { light: "github-light", dark: "github-dark" },
    defaultColor: false,
    decorations,
}));
process.stdout.write(JSON.stringify(rendered));
highlighter.dispose();

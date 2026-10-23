import json
from pathlib import Path
import subprocess

from docutils import nodes
import jedi
from sphinx.errors import ExtensionError


def add_semantic_decorations(blocks):
    lines = []
    for block in blocks:
        first_line = len(lines) + 1
        code_lines = block["code"].split("\n")
        cell_magic = block["code"].lstrip().startswith("%%")
        for line in code_lines:
            lines.append("" if cell_magic or line.lstrip().startswith(("%", "!")) else line)
        lines.append("")

        script = jedi.Script("\n".join(lines))
        for name in script.get_names(all_scopes=True, definitions=True, references=True):
            if name.line < first_line:
                continue
            kinds = {value.type for value in script.infer(name.line, name.column + len(name.name))}
            if len(kinds) != 1:
                continue
            kind = kinds.pop()
            if kind not in {"module", "class", "function"}:
                continue
            line_number = name.line - first_line
            line = code_lines[line_number]
            start = len(line[:name.column].encode("utf-16-le")) // 2
            end = start + len(name.name.encode("utf-16-le")) // 2
            block["decorations"].append({
                "start": {"line": line_number, "character": start},
                "end": {"line": line_number, "character": end},
                "properties": {"class": f"semantic-{kind}"},
            })


def highlight_python(app, doctree, docname):
    if app.builder.format != "html":
        return
    blocks = [
        node for node in doctree.findall(nodes.literal_block)
        if node.get("language") in {"python", "python3", "py", "ipython", "ipython3"}
    ]
    if not blocks:
        return
    rendered_blocks = [{"code": node.astext(), "decorations": []} for node in blocks]
    notebook_inputs = []
    for node, block in zip(blocks, rendered_blocks, strict=True):
        if node.parent.get("nb_element") == "cell_code_source":
            notebook_inputs.append(block)
        else:
            add_semantic_decorations([block])
    add_semantic_decorations(notebook_inputs)
    try:
        result = subprocess.run(
            ["node", str(Path(__file__).with_suffix(".mjs"))],
            input=json.dumps(rendered_blocks),
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        details = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
        raise ExtensionError(f"Shiki highlighting failed; install Node.js and run npm ci.\n{details}") from error
    for node, markup in zip(blocks, json.loads(result.stdout), strict=True):
        container = nodes.container("", classes=["highlight", "notranslate"], ids=node.get("ids", []))
        container += nodes.raw("", markup, format="html")
        node.replace_self(container)


def setup(app):
    app.connect("doctree-resolved", highlight_python)
    app.add_css_file("shiki.css")
    return {"parallel_read_safe": True, "parallel_write_safe": True}

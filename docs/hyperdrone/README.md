# Building the documentation

With Node.js 20 or newer installed and the separate blob repository checked out at
`blob/`, run these commands from this directory using
the repository's `.venv` (create it with `python3 -m venv ../../.venv` if missing):

```bash
npm ci
../../.venv/bin/python -m pip install -r requirements.txt \
  -e "../../interface/python/hyperlux[examples]" \
  -e "../../interface/python/hyperdrone[examples]"
make html
```

Open `_build/html/index.html`, which redirects to HyperLux. MyST-NB executes notebooks headlessly when Sphinx
reads them; subsequent builds reuse unchanged pages. Execution errors fail the build;
cells have no time limit to accommodate scene downloads and native compilation.
Python code is highlighted at build time with Shiki's GitHub Light/Dark themes,
following Furo's light/dark switch; the generated site needs no Node.js or CDN.
Jedi analyzes notebook cells in source order against packages installed in the build
environment to color modules, classes, and functions; unresolved names keep Shiki's
syntax colors. All highlighting works without browser JavaScript.

- `src/*.ipynb`: lean source notebooks, never overwritten by the build.
- `_build/jupyter_execute/src/*.ipynb`: executed notebooks with saved outputs.
- `_build/tmp/`: temporary working directories for generated headers and videos,
  removed after execution. Keeping these inside the checkout lets notebook setup
  cells find the local Python packages. Displayed images and embedded videos remain
  in the executed notebooks and HTML.

Use `make html SPHINXOPTS=-E` to rerun every notebook, including after changing
library code or dependencies that Sphinx does not track. Run `make clean` to remove
generated files. Add new pages to the toctree in `index.rst` and `include_patterns`
in `conf.py`.

`blob/` is an independent Git repository ignored by RLtools. Its `assets/` directory
owns the logos; Sphinx copies them into the website alongside the CSS in `_static/`.

Run `make site` (requires `rsync`) to rebuild the HTML from an empty output directory
and replace `blob/site/` with the successful result, removing obsolete published
files. Caches and temporary files stay in `_build/`; `make clean` leaves `blob/` intact.
`blob/source-commit.txt` records the RLtools commit, suffixed with `-dirty` when tracked
changes are uncommitted. Review and commit the blob repository separately.

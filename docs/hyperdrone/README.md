# Building the documentation

From this directory, using the repository's `.venv` (create it with
`python3 -m venv ../../.venv` if it does not exist):

```bash
../../.venv/bin/python -m pip install -r requirements.txt \
  -e "../../interface/python/hyperlux[examples]" \
  -e "../../interface/python/hyperdrone[examples]"
make html
```

Open `_build/html/index.html`. MyST-NB executes notebooks headlessly when Sphinx
reads them; subsequent builds reuse unchanged pages. Execution errors fail the build;
cells have no time limit to accommodate scene downloads and native compilation.

- `src/*.ipynb`: lean source notebooks, never overwritten by the build.
- `_build/jupyter_execute/src/*.ipynb`: executed notebooks with saved outputs.
- `_build/tmp/`: temporary working directories for generated headers and videos,
  removed after execution. Keeping these inside the checkout lets notebook setup
  cells find the local Python packages. Displayed images and embedded videos remain
  in the executed notebooks and HTML.

Use `make html SPHINXOPTS=-E` to rerun every notebook, including after changing
library code or dependencies that Sphinx does not track. Run `make clean` to remove
generated files. Add new pages to the toctree in `index.rst`.

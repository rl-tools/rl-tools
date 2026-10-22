# Building the documentation

From this directory, using the repository's `.venv` (create it with
`python3 -m venv ../../.venv` if it does not exist):

```bash
../../.venv/bin/python -m pip install -r requirements.txt
make html
```

Open `_build/html/index.html`. Notebooks render their saved outputs without
executing. Add new pages to the toctree in `index.rst`; run `make clean` to
remove generated files.

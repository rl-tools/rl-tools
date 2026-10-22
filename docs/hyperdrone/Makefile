PYTHON ?= ../../.venv/bin/python

.PHONY: html clean

html:
	mkdir -p _build/tmp
	TMPDIR="$(abspath _build/tmp)" $(PYTHON) -m sphinx -M html . _build $(SPHINXOPTS)

clean:
	$(PYTHON) -m sphinx -M clean . _build $(SPHINXOPTS)

PYTHON ?= ../../.venv/bin/python

.PHONY: html site clean

html:
	mkdir -p _build/tmp
	TMPDIR="$(abspath _build/tmp)" $(PYTHON) -m sphinx -M html . _build $(SPHINXOPTS)

site:
	test -e blob/.git
	rm -rf _build/html
	$(MAKE) html
	mkdir -p blob/site
	rsync -a --delete --exclude='/.buildinfo*' _build/html/ blob/site/
	git --no-optional-locks -C ../.. describe --always --abbrev=40 --dirty --match='' > blob/source-commit.txt

clean:
	$(PYTHON) -m sphinx -M clean . _build $(SPHINXOPTS)

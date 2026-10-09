# RepSpace artifact -- common entry points
PY ?= python3

.PHONY: help verify tables test all clean

help:
	@echo "make verify   # artifact integrity + compliance checks + unit tests (no data needed)"
	@echo "make tables   # rebuild paper tables and compare with the published values (no data needed)"
	@echo "make test     # run only the chain-metric and failure-mode regression tests"
	@echo "make all      # verify + tables"

verify:
	$(PY) scripts/verify_artifact.py

tables:
	$(PY) scripts/make_paper_tables.py --check

test:
	$(PY) tests/test_p0_fixes.py

all: verify tables

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

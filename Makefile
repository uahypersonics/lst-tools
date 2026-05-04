# Makefile for lst-tools
# Usage:
#   make lint    # run ruff linter
#   make format  # run ruff formatter
#   make docs    # build MkDocs site
#   make docs-pdf # build MkDocs PDF output
#   make clean   # remove build artifacts
#   make test    # run pytest

SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c
PYTHON ?= .venv/bin/python

.PHONY: lint format docs docs-pdf clean test

lint:
	ruff check src/

format:
	ruff format src/

docs:
	$(PYTHON) -m mkdocs build

docs-pdf:
	$(PYTHON) -m mkdocs build -f mkdocs-pdf.yml

clean:
	rm -rf build dist *.egg-info src/*.egg-info src/**/*.egg-info

test:
	pytest

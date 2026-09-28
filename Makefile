# Makefile for lst-tools
# Usage:
#   make lint    # run ruff linter
#   make format  # run ruff formatter
#   make docs    # build Zensical site
#   make clean   # remove build artifacts
#   make test    # run pytest

SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c
PYTHON ?= .venv/bin/python

.PHONY: lint format docs clean test

lint:
	ruff check src/

format:
	ruff format src/

docs:
	$(PYTHON) -m zensical build --strict

clean:
	rm -rf build dist *.egg-info src/*.egg-info src/**/*.egg-info

test:
	pytest

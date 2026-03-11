# Makefile for lst-tools
# Usage:
#   make lint    # run ruff linter
#   make format  # run ruff formatter
#   make clean   # remove build artifacts
#   make test    # run pytest

SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c

.PHONY: lint format clean test

lint:
	ruff check src/

format:
	ruff format src/

clean:
	rm -rf build dist *.egg-info src/*.egg-info src/**/*.egg-info

test:
	pytest

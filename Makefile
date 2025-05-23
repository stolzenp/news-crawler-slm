# Makefile for setting up and checking the codebase

.PHONY: help install install-dev format lint precommit-init lint-all

help:
	@echo "Makefile commands:"
	@echo "  make install           Install runtime dependencies (requirements.txt)"
	@echo "  make install-dev       Install dev dependencies (requirements-dev.txt)"
	@echo "  make format            Format code using black"
	@echo "  make lint              Lint code using ruff"
	@echo "  make precommit-init    Install pre-commit and set up git hook"
	@echo "  make lint-all          Run format + lint + type checks"

install:
	pip install -r requirements-base.txt
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-base.txt
	pip install -r requirements-dev.txt

format:
	black .

lint:
	ruff .

precommit-init:
	pre-commit install

lint-all: format lint

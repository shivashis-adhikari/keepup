.PHONY: dev check lint fmt typecheck test lab-local example

dev:
	uv venv && uv pip install -e '.[dev]'

check: lint typecheck test

lint:
	ruff check src tests
	ruff format --check src tests

fmt:
	ruff check --fix src tests
	ruff format src tests

typecheck:
	mypy

test:
	pytest

# needs postgres server binaries on the box (initdb, pg_ctl)
lab-local:
	lab/run-local.sh lab/scenarios/basic.toml

example:
	keepup forecast examples/upgraded.toml --out out/upgraded

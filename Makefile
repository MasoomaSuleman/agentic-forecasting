.PHONY: dev lint format

UV := uv run

# Install/sync dev dependencies (run once or after dependency changes).
dev:
	uv sync --group dev

# Format with ruff (replaces black + isort). Writes files in place.
format:
	$(UV) ruff format .
	$(UV) ruff check . --fix --select I

# Full CI mirror: runs the complete pre-commit suite (trailing whitespace,
# end-of-file fixer, YAML/TOML checks, uv-lock, ruff, mypy, nbqa-ruff).
# A passing `make lint` means CI will accept the code.
lint:
	$(UV) pre-commit run --all-files

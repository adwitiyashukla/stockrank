.PHONY: help install install-all test lint fmt data features run run-fast report api app docker clean

PY ?= python
CONFIG ?= configs/default.yaml

help:
	@echo "make install       - install package (core deps)"
	@echo "make install-all   - install with deep learning, api, dashboard, dev extras"
	@echo "make test          - run test suite"
	@echo "make lint          - ruff check"
	@echo "make run           - full research pipeline on the default config"
	@echo "make run-fast      - reduced config, finishes in ~2 minutes on a laptop"
	@echo "make control       - leakage control + signal recovery experiments"
	@echo "make api           - start the FastAPI scoring service"
	@echo "make app           - start the Streamlit research console"

install:
	$(PY) -m pip install -e .

install-all:
	$(PY) -m pip install -e ".[all]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

fmt:
	$(PY) -m ruff format src tests

run:
	$(PY) -m alpha_engine.cli run --config $(CONFIG)

run-fast:
	$(PY) -m alpha_engine.cli run --config configs/fast.yaml

control:
	$(PY) -m alpha_engine.cli run --config configs/leakage_control.yaml
	$(PY) -m alpha_engine.cli run --config configs/signal_recovery.yaml

report:
	$(PY) -m alpha_engine.cli report --config $(CONFIG)

api:
	$(PY) -m uvicorn alpha_engine.api.main:app --reload --port 8000

app:
	$(PY) -m streamlit run dashboard/app.py

docker:
	docker build -f docker/Dockerfile -t alpha-engine:latest .

clean:
	rm -rf artifacts reports/figures/*.png .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

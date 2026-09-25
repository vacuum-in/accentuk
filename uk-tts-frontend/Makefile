PYTHON ?= /home/devops/wiki-stress/ml/.venv/bin/python
export PYTHONPATH := src:.

.PHONY: test test-fast lint prepare serve ready ui up down logs

test:
	$(PYTHON) -m pytest tests -q

test-fast:
	$(PYTHON) -m pytest tests -q -m "not slow"

lint:
	$(PYTHON) -m ruff check src tests

prepare:
	$(PYTHON) -m uktts.cli prepare $(ARGS)

ready:
	$(PYTHON) -m uktts.cli ready

serve:
	$(PYTHON) -m uktts.cli serve --host 0.0.0.0 --port 8000

ui:
	$(PYTHON) -m uktts.gradio_app --api http://127.0.0.1:8000

COMPOSE = docker compose --env-file deploy/.env -f deploy/docker-compose.yml \
	$(if $(wildcard deploy/docker-compose.local.yml),-f deploy/docker-compose.local.yml,)

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail 50

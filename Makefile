.PHONY: help setup run stop clean ingest

help:
	@echo "Available commands:"
	@echo "  make setup    - Initial setup (copy .env.example to .env)"
	@echo "  make run      - Start all services with docker-compose"
	@echo "  make stop     - Stop all services"
	@echo "  make clean    - Remove all containers and volumes"
	@echo "  make ingest   - Run data ingestion only"

setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from template."; \
		echo "Please edit .env with your OpenRouter API key."; \
	else \
		echo ".env file exists - skipping creation."; \
	fi

run: setup
	@command -v docker >/dev/null 2>&1 || { echo "Docker is not installed. Please install Docker first."; exit 1; }
	@docker compose version >/dev/null 2>&1 || { echo "Docker Compose is not available. Please update Docker."; exit 1; }
	docker compose up -d --build
	@echo "Services starting..."
	@echo "Frontend: http://localhost:8600"
	@echo "Backend:  http://localhost:8000"
	@echo "Qdrant:   http://localhost:6333/dashboard"

stop:
	docker compose down

clean:
	docker compose down -v

ingest:
	docker compose run --rm data-ingestion
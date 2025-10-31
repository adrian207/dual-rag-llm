# Makefile for Dual RAG LLM System
# Author: Adrian Johnson <adrian207@gmail.com>

.PHONY: help setup start stop restart logs clean build pull-models rebuild-indexes test health

help:
	@echo "Dual RAG LLM System - Available Commands"
	@echo ""
	@echo "  make setup           - Complete system setup (first time)"
	@echo "  make start           - Start all services"
	@echo "  make stop            - Stop all services"
	@echo "  make restart         - Restart all services"
	@echo "  make logs            - View logs (all services)"
	@echo "  make logs-rag        - View RAG service logs"
	@echo "  make logs-ollama     - View Ollama logs"
	@echo "  make health          - Check service health"
	@echo "  make test            - Run API tests"
	@echo "  make pull-models     - Pull LLM models"
	@echo "  make rebuild-indexes - Rebuild vector indexes"
	@echo "  make build           - Rebuild Docker images"
	@echo "  make clean           - Clean indexes and logs"
	@echo "  make clean-all       - Clean everything (including models)"
	@echo ""

setup:
	@echo "Running automated setup..."
	@bash scripts/setup.sh

start:
	@echo "Starting all services..."
	docker compose up -d
	@echo "Services started. Access Web UI at http://localhost:3000"

stop:
	@echo "Stopping all services..."
	docker compose down

restart:
	@echo "Restarting all services..."
	docker compose restart

logs:
	docker compose logs -f

logs-rag:
	docker compose logs -f rag

logs-ollama:
	docker compose logs -f ollama

logs-webui:
	docker compose logs -f webui

health:
	@echo "Checking service health..."
	@echo ""
	@echo "RAG Service:"
	@curl -s http://localhost:8000/health | python -m json.tool || echo "Not available"
	@echo ""
	@echo "Container Status:"
	@docker compose ps

test:
	@echo "Running API tests..."
	@bash scripts/test_api.sh

pull-models:
	@echo "Pulling LLM models (this takes 30-60 minutes)..."
	docker exec ollama ollama pull qwen2.5-coder:32b-q4_K_M
	docker exec ollama ollama pull deepseek-coder-v2:33b-q4_K_M
	@echo "Models pulled successfully"
	docker exec ollama ollama list

rebuild-indexes:
	@echo "Rebuilding vector indexes..."
	@bash scripts/rebuild_indexes.sh

build:
	@echo "Building Docker images..."
	docker compose build --no-cache

clean:
	@echo "Cleaning indexes and logs..."
	rm -rf rag/indexes/*
	rm -rf rag/logs/*
	@echo "Cleaned. Run 'make rebuild-indexes' to rebuild."

clean-all: clean
	@echo "Removing all data including models..."
	docker compose down -v
	@echo "All data removed. Run 'make setup' to start fresh."

# Development targets
dev-shell:
	docker compose exec rag /bin/bash

dev-logs:
	docker compose logs -f --tail=100 rag

dev-restart-rag:
	docker compose restart rag
	docker compose logs -f rag


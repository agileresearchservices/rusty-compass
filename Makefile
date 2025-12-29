.PHONY: help setup test lint format clean run stop dev dev-api dev-web install-web type-check test-reranker test-hybrid test-query clean-db

# Delegate all make targets to langchain_agent subdirectory
help:
	@cd langchain_agent && $(MAKE) help

setup:
	@cd langchain_agent && $(MAKE) setup

test:
	@cd langchain_agent && $(MAKE) test

test-reranker:
	@cd langchain_agent && $(MAKE) test-reranker

test-hybrid:
	@cd langchain_agent && $(MAKE) test-hybrid

test-query:
	@cd langchain_agent && $(MAKE) test-query

lint:
	@cd langchain_agent && $(MAKE) lint

format:
	@cd langchain_agent && $(MAKE) format

type-check:
	@cd langchain_agent && $(MAKE) type-check

run:
	@cd langchain_agent && $(MAKE) run

stop:
	@cd langchain_agent && $(MAKE) stop

clean:
	@cd langchain_agent && $(MAKE) clean

clean-db:
	@cd langchain_agent && $(MAKE) clean-db

install-web:
	@cd langchain_agent && $(MAKE) install-web

dev-api:
	@cd langchain_agent && $(MAKE) dev-api

dev-web:
	@cd langchain_agent && $(MAKE) dev-web

dev:
	@cd langchain_agent && $(MAKE) dev

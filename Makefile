.PHONY: all clean help venv
.DEFAULT_GOAL := help

MODEL_DIR = model
VENV = .venv
VENV_DIR = ./$(MODEL_DIR)/$(VENV)
REQ_FILE = $(MODEL_DIR)/requirements.txt

POETRY := $(VENV)/bin/poetry

all:
	# dummy

clean:
	rm -rf $(VENV_DIR)

format:
	./$(MODEL_DIR)/format-code.sh

help:
	@echo "Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all                 Dummy target"
	@echo "  clean               Delete Python virtual environment"
	@echo "  format              Format code under: $(MODEL_DIR)/src/"
	@echo "  poetry              Install Python dependencies using Poetry"
	@echo "  init_poetry         Initialize Poetry for existing project"
	@echo "  add_package         Add new package with Poetry use: make add_package PACKAGE=[NAME_OF_PACKAGE]"
	@echo "  venv                Create virtual environment and install Poetry"

poetry:
	cd $(MODEL_DIR) && \
	poetry install && \
	cd ..

init_poetry:
	cd $(MODEL_DIR) && \
	poetry init
	cd ..

add_package:
	@echo "Adding Python package using Poetry..."
	cd $(MODEL_DIR) && $(POETRY) add $(PACKAGE)

venv:
	python -m venv $(VENV_DIR) && \
	source ./$(MODEL_DIR)/.venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r $(REQ_FILE)
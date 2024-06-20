# Makefile for CIFAR-10 Classification Project
export PYTHONPATH := $(shell pwd)
# Variables
ENV_NAME = venv
PYTHON = $(ENV_NAME)/bin/python
PIP = $(ENV_NAME)/bin/pip
PROJECT_DIR = cifar10_classification
DATA_DIR = data
PROCESSED_DATA_DIR = $(DATA_DIR)/processed

# Phony targets
.PHONY: all setup_env install_deps prepare_data extract_features train_model evaluate_model clean

# Default target
all: setup_env install_deps prepare_data extract_features train_model evaluate_model

# Set up virtual environment
setup_env:
	python -m venv $(ENV_NAME)

# Install dependencies
install_deps: setup_env
	$(PIP) install -r requirements.txt

# Prepare data
prepare_data:
	$(PYTHON) $(PROJECT_DIR)/dataset.py

# Extract features
extract_features: prepare_data
	$(PYTHON) $(PROJECT_DIR)/features.py

# Train model
train_model: extract_features
	$(PYTHON) $(PROJECT_DIR)/modeling/train.py

# Evaluate model
evaluate_model: train_model
	$(PYTHON) $(PROJECT_DIR)/modeling/predict.py

grid_search: 
	$(PYTHON) $(PROJECT_DIR)/modeling/grid_search.py
# Clean generated files
clean:
	rm -rf $(PROCESSED_DATA_DIR)/*.npy $(ENV_NAME)


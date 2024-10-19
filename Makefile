# Makefile for managing Python project

# Default target
.PHONY: all
all: format clean

# Format code using black
.PHONY: format
format:
	@echo "Formatting code..."
	black .
	isort .


# Clean Python cache files
.PHONY: clean
clean:
	@echo "Cleaning Python cache files..."
	@echo "Cleaning Python cache files..."
	python -c "import os, shutil; [shutil.rmtree(d) for d in [os.path.join(root, d) for root, dirs, files in os.walk('.') for d in dirs if d == '__pycache__']]"
	python -c "import os; [os.remove(f) for f in [os.path.join(root, f) for root, dirs, files in os.walk('.') for f in files if f.endswith('.pyc') or f.endswith('.pyo')]]"


#!/bin/bash

CODE_DIR="./"

# removing unused imports and variables
find $CODE_DIR -name "*.py" -not -path "*/.venv/*" -exec autoflake --remove-all-unused-imports --remove-unused-variables --in-place {} \;
echo "Autoflake completed."

# formatting code
find $CODE_DIR -name "*.py" -not -path "*/.venv/*" -not -name "config.py" -exec yapf --in-place --style='{based_on_style: google, column_limit=100, indent_width: 4}' {} \;
echo "Yapf completed."

# sorting imports
find $CODE_DIR -name "*.py" -not -path "*/.venv/*" -exec isort {} \;
echo "Isort completed."

echo "Code formatting completed."

#!/usr/bin/env bash
set -e

# Скрипт создания conda-окружения для проекта
# Использование:
#   bash setup_env.sh

ENV_NAME="scinti-clustering"

echo "Creating conda environment '${ENV_NAME}' from environment.yml ..."
conda env remove -n "${ENV_NAME}" -y || true
conda env create -f environment.yml

echo "Environment '${ENV_NAME}' created."
echo "Activate it with:"
echo "  conda activate ${ENV_NAME}"

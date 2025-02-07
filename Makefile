.PHONY: train test lint format clean

train:
	python scripts/train.py --model graphsage --epochs 100

train-gat:
	python scripts/train.py --model gat --epochs 100

train-gin:
	python scripts/train.py --model gin --epochs 100

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf .ruff_cache/

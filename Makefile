.PHONY: install test notebook clean

install:
	pip install -e ".[dev]"

test:
	pytest

notebook:
	jupytext --to ipynb notebooks/04_reason_codes.py -o notebooks/04_reason_codes.ipynb
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=credit-venv --ExecutePreprocessor.timeout=600 notebooks/04_reason_codes.ipynb

clean:
	rm -rf .pytest_cache __pycache__ src/*.egg-info

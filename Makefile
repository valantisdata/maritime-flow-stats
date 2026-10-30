.PHONY: install test lint demo red-sea piraeus db dashboard clean

install:
	pip install -e ".[dev,app]"

test:
	pytest --cov=maritime-flow --cov-report=term-missing

lint:
	ruff check src tests

demo:
	maritime-flow demo --out reports/demo

red-sea:
	maritime-flow -v analyze --target "Suez" --comparator "Good Hope" --event red_sea --out reports/red-sea

piraeus:
	maritime-flow -v analyze --kind port --target "Piraeus" --event red_sea --out reports/piraeus

db:
	maritime-flow db load --demo

dashboard:
	maritime-flow dashboard

clean:
	rm -rf reports .pytest_cache .ruff_cache build dist *.egg-info

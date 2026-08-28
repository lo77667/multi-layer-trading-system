.PHONY: install test sample backtest preflight

install:
	python3 -m pip install -e '.[dev]'

test:
	python3 -m pytest -q

sample:
	python3 -m trading_system.cli generate-sample --output data/sample/eurusd.csv

backtest: sample
	python3 -m trading_system.cli backtest --csv data/sample/eurusd.csv --symbol EUR_USD

preflight:
	python3 -m trading_system.cli preflight

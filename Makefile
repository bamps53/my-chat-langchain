.PHONY: start
start:
	uvicorn main:app --reload --port 10000

.PHONY: format
format:
	black .
	isort .
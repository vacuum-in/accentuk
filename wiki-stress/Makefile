.PHONY: setup lint typecheck test test-race build download parse validate migrate import publish rollback benchmark export clean api

setup:
	uv --directory etl sync --all-groups

lint:
	uv --directory etl run ruff check src tests
	cd api && go vet ./...

typecheck:
	uv --directory etl run mypy src tests

test:
	uv --directory etl run pytest tests
	cd api && go test ./...

test-race:
	cd api && go test -race ./...

build:
	cd api && go build -trimpath -buildvcs=false -o bin/stress-api ./cmd/stress-api

download parse validate migrate import publish rollback benchmark export:
	uv --directory etl run ukstress $@

api:
	cd api && go run ./cmd/stress-api

clean:
	rm -rf etl/.pytest_cache etl/.mypy_cache etl/.ruff_cache etl/.venv api/bin output

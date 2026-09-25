.PHONY: fmt vet test test-race test-integration build

fmt:
	gofmt -w $$(find . -name '*.go' -type f)

vet:
	go vet ./...

test:
	go test ./...

test-race:
	go test -race ./...

# Requires UKSTRESS_TEST_DATABASE_URL (a throwaway PostgreSQL database);
# skipped entirely otherwise. -p 1 is required, not optional: the
# repository and httpapi packages both mutate the single global
# active_dataset row against the same live database, and Go's default
# per-package test parallelism would run their test binaries as
# concurrent, racing OS processes. See internal/testdb for details.
test-integration:
	go test -p 1 ./...

build:
	go build -trimpath -buildvcs=false -o bin/stress-api ./cmd/stress-api


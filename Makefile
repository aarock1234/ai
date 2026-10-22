-include .env
export

GOLANGCI_LINT_VERSION := v2.15.0
GOLANGCI_LINT := go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)

# MODULES lists the root module and the nested modules that isolate optional
# dependencies (OpenTelemetry, the JavaScript interpreter, the terminal UI,
# the WebAssembly runtime of OPA policies and the shell interpreter of the
# in-memory sandbox).
MODULES := . otel codemode tui policyopawasm sandboxshell

.PHONY: build test lint format format-check fix check check-static

build:
	@for module in $(MODULES); do \
		echo "building $$module"; \
		(cd $$module && go build ./...) || exit 1; \
	done

test:
	go test -race ./...

test-integration:
	go test -race -tags=integration ./...

lint:
	@for module in $(MODULES); do \
		echo "linting $$module"; \
		(cd $$module && $(GOLANGCI_LINT) run --tests=false) || exit 1; \
	done

format:
	@for module in $(MODULES); do \
		(cd $$module && go fmt ./...) || exit 1; \
	done

format-check:
	@for module in $(MODULES); do \
		echo "checking format of $$module"; \
		(cd $$module && $(GOLANGCI_LINT) fmt --diff) || exit 1; \
	done

check-static: build lint format-check

fix:
	go fix ./...

check: format
	go vet ./...
	go test -race ./...
	gofmt -l .

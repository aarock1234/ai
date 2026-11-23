GOLANGCI_LINT_VERSION := v2.15.0
GOLANGCI_LINT := go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)

# MODULES lists the root module and the nested modules that isolate optional
# dependencies (OpenTelemetry, the JavaScript interpreter, the terminal UI,
# the WebAssembly runtime of OPA policies and the shell interpreter of the
# in-memory sandbox). Dependabot cannot read this list: keep the directories
# in .github/dependabot.yml in step with it.
MODULES := . otel codemode tui policyopawasm sandboxshell

# for-each-module runs a command in every module directory and stops at the
# first failure.
define for-each-module
	@for module in $(MODULES); do \
		echo "$(1): $$module"; \
		(cd $$module && $(2)) || exit 1; \
	done
endef

.PHONY: build test test-integration lint format format-check vet fix check check-static workspace

# workspace joins the modules so that the nested ones build against the
# working core instead of the published version their go.mod requires.
# go.work is untracked.
workspace:
	@test -f go.work || go work init $(MODULES)

build:
	$(call for-each-module,building,go build ./...)

test:
	go test -race ./...

test-integration:
	go test -race -tags=integration ./...

lint:
	$(call for-each-module,linting,$(GOLANGCI_LINT) run --tests=false)

format:
	$(call for-each-module,formatting,go fmt ./...)

format-check:
	$(call for-each-module,checking format,$(GOLANGCI_LINT) fmt --diff)

vet:
	$(call for-each-module,vetting,go vet ./...)

fix:
	$(call for-each-module,fixing,go fix ./...)

check-static: build lint format-check

check: format vet test

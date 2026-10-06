CARGO ?= cargo
RUSTC ?= rustc
PYTHON ?= python3
JOBS ?= 2
CARGO_TARGET_DIR ?= target
# Resolve the native target only when a static recipe runs (GNU/BSD make).
STATIC_TARGET ?= $$($(RUSTC) -vV | sed -n 's/^host: //p')
BINDIR ?= /data/scripts
DESTDIR ?=
TEST_ARGS ?= --test-threads=1
ARGS ?=

CARGO_OPTIONS = --locked --jobs $(JOBS) --target-dir "$(CARGO_TARGET_DIR)"

all: release
build: release

.PHONY: all build release debug static install install-static demo run examples \
	grid edge test-w0 test test-rust test-python test-doc test-lib test-st \
	test-ignored test-all test-rust-all fmt fmt-check vet lint check verify \
	doc clean chartgen plot gallery help

release:
	@$(CARGO) build $(CARGO_OPTIONS) --release

debug:
	@$(CARGO) build $(CARGO_OPTIONS)

static:
	@target="$(STATIC_TARGET)"; \
		flags="$(RUSTFLAGS) -C target-feature=+crt-static"; \
		case "$$target" in \
			*-freebsd) flags="$$flags -C link-arg=-Wno-unused-command-line-argument" ;; \
		esac; \
		RUSTFLAGS="$$flags" $(CARGO) build $(CARGO_OPTIONS) --release --target "$$target"
	@kind="$$(LC_ALL=C file -b "$(CARGO_TARGET_DIR)/$(STATIC_TARGET)/release/tet")"; \
		case "$$kind" in \
			*"statically linked"*|*"static-pie linked"*) ;; \
			*) printf '%s\n' "error: not a static executable: $$kind" >&2; exit 1 ;; \
		esac

install: release
	@install -d "$(DESTDIR)$(BINDIR)"
	@install -m 0755 "$(CARGO_TARGET_DIR)/release/tet" "$(DESTDIR)$(BINDIR)/tet"

install-static: static
	@install -d "$(DESTDIR)$(BINDIR)"
	@install -m 0755 "$(CARGO_TARGET_DIR)/$(STATIC_TARGET)/release/tet" "$(DESTDIR)$(BINDIR)/tet"

demo: release
	@"$(CARGO_TARGET_DIR)/release/tet" 50 1.3 0.1 0.5 0

run: release
	@"$(CARGO_TARGET_DIR)/release/tet" $(ARGS)

examples:
	@$(CARGO) build $(CARGO_OPTIONS) --release --examples

grid:
	@$(CARGO) run $(CARGO_OPTIONS) --release --example grid_runner -- $(ARGS)

edge:
	@$(CARGO) run $(CARGO_OPTIONS) --release --example edge_runner -- $(ARGS)

test-w0:
	@$(CARGO) run $(CARGO_OPTIONS) --release --example test_w0

test: test-rust test-python test-doc

test-rust:
	@$(CARGO) test $(CARGO_OPTIONS) --release --all-targets -- $(TEST_ARGS)

test-python:
	@PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

test-doc:
	@$(CARGO) test $(CARGO_OPTIONS) --release --doc -- $(TEST_ARGS)

test-lib:
	@$(CARGO) test $(CARGO_OPTIONS) --release --lib -- $(TEST_ARGS)

test-st:
	@env -u TET_MT -u RAYON_NUM_THREADS \
		$(CARGO) test $(CARGO_OPTIONS) --release --all-targets -- $(TEST_ARGS)

test-ignored:
	@$(CARGO) test $(CARGO_OPTIONS) --release --all-targets -- --ignored $(TEST_ARGS)

test-all: test-rust-all test-python test-doc

test-rust-all:
	@$(CARGO) test $(CARGO_OPTIONS) --release --all-targets -- --include-ignored $(TEST_ARGS)

fmt:
	@$(CARGO) fmt --all

fmt-check:
	@$(CARGO) fmt --all -- --check

vet:
	@$(CARGO) clippy $(CARGO_OPTIONS) --release --all-targets -- -D warnings

lint: vet

check:
	@$(CARGO) check $(CARGO_OPTIONS) --all-targets

verify: fmt-check vet test

doc:
	@$(CARGO) doc $(CARGO_OPTIONS) --release --no-deps

clean:
	@$(CARGO) clean --target-dir "$(CARGO_TARGET_DIR)"

chartgen: release
	@CARGO_TARGET_DIR="$(CARGO_TARGET_DIR)" bash scripts/chartgen.sh $(ARGS)

plot:
	@$(PYTHON) scripts/plot3d.py $(ARGS)

gallery:
	@bash scripts/chartgallery.sh

help:
	@printf '%s\n' \
		'make / all / build / release  Build the optimized tet executable' \
		'debug                        Build the debug executable' \
		'static                       Build a fully static native executable' \
		'install / install-static     Install tet into BINDIR (default /data/scripts)' \
		'demo                         Run one 50-digit complex-base example' \
		'run ARGS="..."               Run tet with supplied CLI arguments' \
		'examples                     Build all three example programs' \
		'grid / edge ARGS="..."       Run the grid or edge-case TSV sweep' \
		'test-w0                      Run the Lambert W diagnostic example' \
		'test                         Run default Rust, Python and documentation tests' \
		'test-lib / test-rust          Run library-only / all-target Rust tests' \
		'test-python / test-doc        Run Python / documentation tests' \
		'test-st                      Run Rust tests with MT settings unset' \
		'test-ignored / test-all       Run ignored Rust tests / the entire battery' \
		'fmt / fmt-check              Apply / check Rust formatting' \
		'vet / lint                   Run all-target Clippy with warnings denied' \
		'check / verify               Compile-check / run formatting, lint and tests' \
		'doc                          Build crate documentation without dependencies' \
		'chartgen ARGS="..."          Generate a CSV sweep with scripts/chartgen.sh' \
		'plot ARGS="..."              Render CSV data with scripts/plot3d.py' \
		'gallery                      Regenerate the gallery from existing CSVs' \
		'clean                        Remove Cargo outputs only; preserve chart data' \
		'Variables: CARGO, RUSTC, PYTHON, JOBS, CARGO_TARGET_DIR, STATIC_TARGET,' \
		'           BINDIR, DESTDIR, TEST_ARGS, ARGS; runtime TET_* settings are inherited.'

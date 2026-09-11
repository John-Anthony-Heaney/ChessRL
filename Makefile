# ChessRL -- zero-dependency C engine + population RL trainer.

CC      ?= cc
ARCH    := $(shell $(CC) -mcpu=native -E -x c /dev/null >/dev/null 2>&1 && echo -mcpu=native)
CFLAGS  ?= -O3 -std=c11 -D_DARWIN_C_SOURCE -Wall -Wextra -Wno-unused-parameter -funroll-loops \
           -fno-math-errno -ffp-contract=fast $(ARCH) -Isrc
LDFLAGS ?= -lm -lpthread
BUILD   := build

CORE_SRC := src/chess.c src/net.c src/arena.c src/search.c
CORE_OBJ := $(patsubst src/%.c,$(BUILD)/%.o,$(CORE_SRC))

.PHONY: all clean lib test integration bench fast debug app
all: $(BUILD)/chessrl lib

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/chessrl: $(CORE_OBJ) $(BUILD)/train.o $(BUILD)/uci.o $(BUILD)/main.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Shared library for the Python/ctypes front-end.
lib: $(BUILD)/libchessrl.dylib
$(BUILD)/libchessrl.dylib: $(CORE_SRC) src/api.c src/train.c
	$(CC) $(CFLAGS) -fPIC -shared $^ -o $@ $(LDFLAGS)

# --------------------------------------------------------------------- tests
test: $(BUILD)/test_perft $(BUILD)/test_rules $(BUILD)/test_net $(BUILD)/test_search
	@echo "=== perft  ===" && $(BUILD)/test_perft
	@echo "=== rules  ===" && $(BUILD)/test_rules
	@echo "=== net    ===" && $(BUILD)/test_net
	@echo "=== search ===" && $(BUILD)/test_search

# Full-stack test: builds the shared library, starts the server, drives every endpoint.
integration: lib
	python3 tests/test_integration.py

$(BUILD)/test_perft: tests/test_perft.c $(CORE_OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
$(BUILD)/test_rules: tests/test_rules.c $(CORE_OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
$(BUILD)/test_net: tests/test_net.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -O1 $^ -o $@ $(LDFLAGS)
$(BUILD)/test_search: tests/test_search.c $(CORE_OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# ------------------------------------------------------------- macOS app
# A real .app bundle: AppKit + Core Graphics in Swift, linked directly against
# the engine objects.  No browser, no server, no Python.
SWIFTC    ?= swiftc
APP       := $(BUILD)/ChessRL.app
APP_SRC   := $(wildcard mac/*.swift)
SWIFTFLAGS ?= -O -warnings-as-errors -import-objc-header mac/bridge.h

app: $(APP)

$(APP): $(APP_SRC) mac/bridge.h mac/Info.plist $(CORE_OBJ) $(BUILD)/api.o
	@mkdir -p $(APP)/Contents/MacOS $(APP)/Contents/Resources
	$(SWIFTC) $(SWIFTFLAGS) $(APP_SRC) $(CORE_OBJ) $(BUILD)/api.o \
	    -o $(APP)/Contents/MacOS/ChessRL
	@cp mac/Info.plist $(APP)/Contents/Info.plist
	@printf 'APPL????' > $(APP)/Contents/PkgInfo
	@codesign --force --sign - $(APP) >/dev/null 2>&1 || true
	@touch $(APP)
	@echo "built $(APP)"

bench: $(BUILD)/chessrl
	$(BUILD)/chessrl bench

debug: CFLAGS := -O0 -g -std=c11 -D_DARWIN_C_SOURCE -Wall -Wextra -Wno-unused-parameter -fsanitize=address,undefined -Isrc
debug: clean $(BUILD)/chessrl

clean:
	rm -rf $(BUILD)

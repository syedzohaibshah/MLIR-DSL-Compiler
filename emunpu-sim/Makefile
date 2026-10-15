# Standalone build for the emulated multi-core NPU simulator.
#
# No dependencies beyond a C compiler and the C standard library (libm).
#   make            # build the shared object, static lib, tests, and demos
#   make shared     # build just the shared object (libemunpu.so)
#   make static     # build just the static archive (libemunpu.a)
#   make test       # build + run the functional tests (against the shared lib)
#   make demo       # build + run every demo (against the shared lib)
#   make viz-demo   # print the two-terminal visualizer instructions
#   make clean
#
# The shared object exports only the EMUNPU_API device/runtime API; internal
# helpers stay private (built with -fvisibility=hidden).

CC      ?= cc
CFLAGS  ?= -O2 -g -Wall -Wextra -std=c11
CFLAGS  += -Iinclude
LDLIBS  += -lm

# Shared-object soname/version (mirrors the CMake VERSION/SOVERSION).
SOVERSION := 1
VERSION   := 1.0.0

BUILD    := build
LIB_SRCS := $(wildcard src/*.c)
LIB_OBJS := $(patsubst src/%.c,$(BUILD)/%.o,$(LIB_SRCS))
PIC_OBJS := $(patsubst src/%.c,$(BUILD)/%.pic.o,$(LIB_SRCS))

STATIC   := $(BUILD)/libemunpu.a
SONAME   := libemunpu.so.$(SOVERSION)
SHARED   := $(BUILD)/libemunpu.so.$(VERSION)

TESTS := device_test ops_test
DEMOS := mlp_demo cnn_demo transformer_demo lowlevel_demo
TEST_BINS := $(addprefix $(BUILD)/,$(TESTS))
DEMO_BINS := $(addprefix $(BUILD)/,$(DEMOS))

.PHONY: all shared static test demo viz-demo clean
all: shared static $(TEST_BINS) $(DEMO_BINS)
shared: $(SHARED)
static: $(STATIC)

# Static-archive objects.
$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

# Position-independent objects for the shared object; hide non-API symbols.
$(BUILD)/%.pic.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) -fPIC -fvisibility=hidden -DEMUNPU_BUILDING_SHARED -c $< -o $@

$(STATIC): $(LIB_OBJS)
	$(AR) rcs $@ $^

$(SHARED): $(PIC_OBJS)
	$(CC) -shared -Wl,-soname,$(SONAME) $^ $(LDLIBS) -o $@
	ln -sf $(notdir $(SHARED)) $(BUILD)/$(SONAME)
	ln -sf $(SONAME) $(BUILD)/libemunpu.so

# Executables link against the shared object (rpath so they run in-tree).
$(TEST_BINS): $(BUILD)/%: test/%.c $(SHARED) | $(BUILD)
	$(CC) $(CFLAGS) $< -L$(BUILD) -lemunpu -Wl,-rpath,'$$ORIGIN' $(LDLIBS) -o $@

$(DEMO_BINS): $(BUILD)/%: demo/%.c $(SHARED) | $(BUILD)
	$(CC) $(CFLAGS) $< -L$(BUILD) -lemunpu -Wl,-rpath,'$$ORIGIN' $(LDLIBS) -o $@

$(BUILD):
	mkdir -p $(BUILD)

# Each demo self-validates and exits non-zero on mismatch, so `make test`
# running them is a real end-to-end check, not just a smoke test.
test: $(TEST_BINS) $(DEMO_BINS)
	@set -e; for t in $(TEST_BINS) $(DEMO_BINS); do \
	  echo "== $$t"; ./$$t; done

demo: $(DEMO_BINS)
	@set -e; for d in $(DEMO_BINS); do echo "== $$d"; ./$$d; done

viz-demo: $(BUILD)/mlp_demo
	@echo "Two terminals:"
	@echo "  1) python demo/visualizer.py /tmp/emunpu.events"
	@echo "  2) IREE_EMUNPU_VIZ=/tmp/emunpu.events ./$(BUILD)/mlp_demo"

clean:
	rm -rf $(BUILD)

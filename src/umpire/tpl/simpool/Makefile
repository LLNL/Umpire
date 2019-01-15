# config.mk can overwrite these defaults
-include config.mk

CXX ?= clang++
CXXFLAGS ?= -O1 -g
LDFLAGS ?=

# Defines for testing
ifeq ($(HAS_CUDA),YES)

CXXFLAGS += -I/usr/local/cuda/include -DHAS_CUDA
LDFLAGS += -L/usr/local/cuda/lib64 -lcudart

ifeq ($(USE_UVM_ALLOC),YES)
CXXFLAGS += -DUSE_UVM_ALLOC
endif

endif

HEADERS=StdAllocator.hpp \
	FixedSizePool.hpp \
	DynamicSizePool.hpp

TESTS=$(patsubst %.cpp, %, $(wildcard test-*.cpp))

all: $(TESTS)

$(TESTS) : % : %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o $@

.PHONY: check
check: $(TESTS)
	@for f in $(TESTS); do \
		printf "\033[34mRunning %s... \033[0m" $$f; \
		./$$f &> /dev/null; \
		if [ ! $$? ]; then \
			printf '\033[31mFAILED\033[0m\n'; \
		else \
			printf '\033[32;1mOK\033[0m\n'; \
		fi; \
	done

.PHONY: clean
clean:
	rm -f $(TESTS)

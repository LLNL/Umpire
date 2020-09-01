gcc:
ifeq ($(DEBUG),1)
	DOCKER_BUILDKIT=1 docker build --target gcc --no-cache --progress plain .
else
	DOCKER_BUILDKIT=1 docker build --target gcc --no-cache .
endif

clang:
ifeq ($(DEBUG),1)
	DOCKER_BUILDKIT=1 docker build --target clang --no-cache --progress plain .
else
	DOCKER_BUILDKIT=1 docker build --target clang --no-cache .
endif

nvcc:
ifeq ($(DEBUG),1)
	DOCKER_BUILDKIT=1 docker build --target nvcc --no-cache --progress plain .
else
	DOCKER_BUILDKIT=1 docker build --target nvcc --no-cache .
endif

hcc:
ifeq ($(DEBUG),1)
	DOCKER_BUILDKIT=1 docker build --target hcc --no-cache --progress plain .
else
	DOCKER_BUILDKIT=1 docker build --target hcc --no-cache .
endif

hip:
ifeq ($(DEBUG),1)
	DOCKER_BUILDKIT=1 docker build --target hip --no-cache --progress plain .
else
	DOCKER_BUILDKIT=1 docker build --target hip --no-cache .
endif

sycl:
ifeq ($(DEBUG),1)
	DOCKER_BUILDKIT=1 docker build --target sycl --no-cache --progress plain .
else
	DOCKER_BUILDKIT=1 docker build --target sycl --no-cache .
endif

help:
	@echo 'usage: make [variable] [target]'
	@echo ''
	@echo 'Build Umpire using Docker!'
	@echo ''
	@echo 'target:'
	@echo '    gcc                            build with GCC 8'
	@echo '    clang                          build with clang 6'
	@echo '    nvcc                           build with CUDA 9'
	@echo '    hcc                            build with hcc'
	@echo '    hip                            build with HIP'
	@echo ''
	@echo 'variable:'
	@echo '    DEBUG                          display all output if set to 1'
	@echo ''
	@echo 'For example'
	@echo ''
	@echo '    make DEBUG=1 nvcc'
	@echo ''
	@echo 'builds with the nvcc Docker image and displays all output.'

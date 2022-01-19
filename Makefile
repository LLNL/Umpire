ifeq ($(DEBUG),1)
	DebugArgs=--progress plain
else
	DebugArgs=
endif

gcc7:
	DOCKER_BUILDKIT=1 docker build --target gcc7 --no-cache $(DebugArgs) .

gcc8:
	DOCKER_BUILDKIT=1 docker build --target gcc8 --no-cache $(DebugArgs) .

gcc9:
	DOCKER_BUILDKIT=1 docker build --target gcc9 --no-cache $(DebugArgs) .

gcc11:
	DOCKER_BUILDKIT=1 docker build --target gcc11 --no-cache $(DebugArgs) .

clang11:
	DOCKER_BUILDKIT=1 docker build --target clang11 --no-cache $(DebugArgs) .

clang12:
	DOCKER_BUILDKIT=1 docker build --target clang12 --no-cache $(DebugArgs) .

clang13:
	DOCKER_BUILDKIT=1 docker build --target clang13 --no-cache $(DebugArgs) .

nvcc10:
	DOCKER_BUILDKIT=1 docker build --target nvcc10 --no-cache $(DebugArgs) .

nvcc11:
	DOCKER_BUILDKIT=1 docker build --target nvcc11 --no-cache $(DebugArgs) .

hip:
	DOCKER_BUILDKIT=1 docker build --target hip --no-cache $(DebugArgs) .

sycl:
	DOCKER_BUILDKIT=1 docker build --target sycl --no-cache $(DebugArgs) .

style:
	scripts/docker/apply-style.sh

help:
	@echo 'usage: make [variable] [target]'
	@echo ''
	@echo 'Build Umpire using Docker!'
	@echo ''
	@echo 'target:'
	@echo '    gccN                           build with GCC N'
	@echo '    clangN                         build with clang N'
	@echo '    nvccN                          build with CUDA N'
	@echo '    hip                            build with HIP'
	@echo ''
	@echo 'variable:'
	@echo '    DEBUG                          display all output if set to 1'
	@echo ''
	@echo 'For example'
	@echo ''
	@echo '    make DEBUG=1 nvcc11'
	@echo ''
	@echo 'builds with the nvcc Docker image for nvcc 11.1.1 and displays all output.'

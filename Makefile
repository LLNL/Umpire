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

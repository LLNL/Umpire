blt_add_executable(
  NAME malloc
  SOURCES malloc.cxx
  DEPENDS_ON 
    umpire
    )

if (ENABLE_CUDA)
  blt_add_executable(
    NAME malloc_gpu
    SOURCES malloc_gpu.cxx
    DEPENDS_ON
      umpire
      cuda)
endif ()

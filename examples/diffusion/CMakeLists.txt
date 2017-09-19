if (ENABLE_CUDA)
  blt_add_executable(
    NAME diffusion
    SOURCES diffusion.cpp
    DEPENDS_ON 
      umpire cuda)
endif ()

if (ENABLE_CUDA)
  blt_add_executable(
    NAME recipe_one_shrink
    SOURCES recipe_one_shrink.cpp
    DEPENDS_ON 
      umpire)
endif ()

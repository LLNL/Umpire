if (ENABLE_CUDA)
  blt_add_executable(
    NAME recipe_one_shrink
    SOURCES recipe_one_shrink.cpp
    DEPENDS_ON 
      umpire)
endif ()

blt_add_executable(
  NAME recipe_two_no_introspection
  SOURCES recipe_two_no_introspection.cpp
  DEPENDS_ON umpire)

if (ENABLE_CUDA)
  blt_add_executable(
    NAME recipe_three_advise_pool
    SOURCES recipe_three_advise_pool.cpp
    DEPENDS_ON umpire)
endif()

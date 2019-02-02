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
    NAME recipe_three_pool_advice
    SOURCES recipe_three_pool_advice.cpp
    DEPENDS_ON umpire)
endif()

blt_add_executable(
  NAME recipe_dynamic_pool_heuristic
  SOURCES recipe_dynamic_pool_heuristic.cpp
  DEPENDS_ON umpire)

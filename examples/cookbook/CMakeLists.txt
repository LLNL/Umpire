if (ENABLE_CUDA)
  blt_add_executable(
    NAME recipe_shrink
    SOURCES recipe_shrink.cpp
    DEPENDS_ON 
      umpire)
endif ()

blt_add_executable(
  NAME recipe_no_introspection
  SOURCES recipe_no_introspection.cpp
  DEPENDS_ON umpire)

if (ENABLE_CUDA)
  blt_add_executable(
    NAME recipe_pool_advice
    SOURCES recipe_pool_advice.cpp
    DEPENDS_ON umpire)
endif()

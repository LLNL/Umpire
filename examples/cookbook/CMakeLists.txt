blt_add_executable(
  NAME recipe_no_introspection
  SOURCES recipe_no_introspection.cpp
  DEPENDS_ON umpire)

if (ENABLE_NUMA)
  blt_add_executable(
    NAME recipe_numa_allocator
    SOURCES recipe_numa_allocator.cpp
    DEPENDS_ON umpire)
endif ()

if (ENABLE_CUDA)
  blt_add_executable(
    NAME recipe_shrink
    SOURCES recipe_shrink.cpp
    DEPENDS_ON
      umpire)

  blt_add_executable(
    NAME recipe_advice_device_id
    SOURCES recipe_advice_device_id.cpp
    DEPENDS_ON
      umpire)

  blt_add_executable(
    NAME recipe_pool_advice
    SOURCES recipe_pool_advice.cpp
    DEPENDS_ON umpire)

  blt_add_executable(
    NAME recipe_move_to_managed
    SOURCES recipe_move_to_managed.cpp
    DEPENDS_ON umpire)
endif ()

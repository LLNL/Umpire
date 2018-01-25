blt_add_executable(
  NAME malloc
  SOURCES malloc.cxx
  DEPENDS_ON 
    umpire)

blt_add_executable(
  NAME strategy_example
  SOURCES strategy_example.cpp
  DEPENDS_ON
    umpire)

if (ENABLE_FORTRAN)
  blt_add_executable(
    NAME allocator_c
    SOURCES allocator.c
    DEPENDS_ON
      umpire
  )
endif ()

blt_add_executable(
  NAME vector_allocator
  SOURCES vector_allocator.cpp
  DEPENDS_ON 
    umpire)

add_subdirectory(lulesh)

add_subdirectory(diffusion)

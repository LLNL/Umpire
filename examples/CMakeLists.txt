blt_add_executable(
  NAME malloc
  SOURCES malloc.cxx
  DEPENDS_ON 
    umpire
    umpire_space
    umpire_alloc
    )

blt_add_executable(
  NAME allocator_c
  SOURCES allocator.c
  DEPENDS_ON
    umpire
)

add_subdirectory(lulesh)

add_subdirectory(diffusion)

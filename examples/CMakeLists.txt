blt_add_executable(
  NAME malloc
  SOURCES malloc.cxx
  DEPENDS_ON 
    umpire
    umpire_space
    umpire_alloc
    )

add_subdirectory(lulesh)

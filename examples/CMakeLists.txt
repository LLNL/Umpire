blt_add_executable(
  NAME malloc
  SOURCES malloc.cxx
  DEPENDS_ON 
    umpire
    )

add_subdirectory(lulesh)

set (umpire_sources
  ResourceManager.cpp)

set (umpire_headers
    MemorySpace.hpp
    ResourceManager.hpp
    Types.hpp)

blt_add_library(
  NAME umpire
  HEADERS ${umpire_headers}
  SOURCES ${umpire_sources})

add_subdirectory(tests)

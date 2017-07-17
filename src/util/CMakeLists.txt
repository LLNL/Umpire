set (umpire_util_headers
  Exception.hpp
  Macros.hpp)

set (umpire_util_sources
  Exception.cpp)

blt_add_library(
  NAME umpire_util
  HEADERS ${umpire_util_headers}
  SOURCES ${umpire_util_sources}
  HEADERS_OUTPUT_SUBDIR umpire/util
)

install(TARGETS
  umpire_util
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

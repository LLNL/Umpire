set (umpire_strategy_headers
  AllocationStrategy.hpp
  Pool.hpp)

set (umpire_stategy_sources
  AllocationStrategy.cpp
  Pool.cpp)

set (umpire_strategy_depends
  umpire)

blt_add_library(
  NAME umpire_strategy
  HEADERS ${umpire_strategy_headers}
  SOURCES ${umpire_stategy_sources}
  HEADERS_OUTPUT_SUBDIR umpire/strategy
  DEPENDS_ON ${umpire_strategy_depends})

install(TARGETS
  umpire_strategy
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

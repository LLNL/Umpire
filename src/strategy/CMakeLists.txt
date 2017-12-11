set (umpire_strategy_headers
  AllocationStrategy.hpp
  AllocationStrategyFactory.hpp
  AllocationStrategyRegistry.hpp
  DefaultAllocationStrategy.hpp
  GenericAllocationStrategyFactory.hpp
  GenericAllocationStrategyFactory.inl
  MonotonicAllocationStrategy.hpp
  Pool.hpp
  SimpoolAllocationStrategy.hpp)

set (umpire_stategy_sources
  AllocationStrategyRegistry.cpp
  DefaultAllocationStrategy.cpp
  MonotonicAllocationStrategy.cpp
  Pool.cpp
  SimpoolAllocationStrategy.cpp)

set (umpire_strategy_depends
  umpire
  umpire_util
  umpire_tpl_simpool)

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

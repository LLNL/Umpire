set (simpool_headers
  DynamicSizePool.hpp
  FixedSizePool.hpp
  StdAllocator.hpp)

set (simpool_sources
  DynamicSizePool.cpp)

blt_add_library(
  NAME umpire_tpl_simpool
  SOURCES ${simpool_sources}
  HEADERS ${simpool_headers}
  DEPENDS_ON umpire_util)

target_link_libraries(umpire_tpl_simpool INTERFACE)

target_include_directories(
  umpire_tpl_simpool
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(FILES
  ${simpool_headers}
  DESTINATION include/umpire/tpl/simpool)

install(TARGETS
  umpire_tpl_simpool
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

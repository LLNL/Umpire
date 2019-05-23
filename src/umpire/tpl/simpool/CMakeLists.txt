set (simpool_headers
  DynamicSizePool.hpp
  FixedSizePool.hpp
  StdAllocator.hpp)

blt_add_library(
  NAME umpire_tpl_simpool
  HEADERS ${simpool_headers})

target_link_libraries(umpire_tpl_simpool INTERFACE)

target_include_directories(
  umpire_tpl_simpool
  INTERFACE
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

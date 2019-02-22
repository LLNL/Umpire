set (simpool_headers
  DynamicSizePool.hpp
  FixedSizePool.hpp
  StdAllocator.hpp)

blt_add_library(
  NAME umpire_tpl_simpool
  HEADERS ${simpool_headers})

target_link_libraries(umpire_tpl_simpool INTERFACE)

install(FILES
  ${simpool_headers}
  DESTINATION include/umpire/tpl/simpool)

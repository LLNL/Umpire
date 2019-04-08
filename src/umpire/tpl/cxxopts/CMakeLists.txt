set (cxxopts_headers
  ./cxxopts/include/cxxopts.hpp )

blt_add_library(
  NAME umpire_tpl_cxxopts
  HEADERS ${cxxopts_headers})

target_link_libraries(umpire_tpl_cxxopts INTERFACE)

install(FILES
  ${cxxopts_headers}
  DESTINATION include/umpire/tpl/cxxopts/cxxopts/include)

install(TARGETS
  umpire_tpl_cxxopts
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

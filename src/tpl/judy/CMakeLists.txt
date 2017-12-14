blt_add_library(
  NAME umpire_tpl_judy
  SOURCES judy.c
  HEADERS judy.h judyLArray.h
  HEADERS_OUTPUT_SUBDIR umpire/tpl/judy)

install(TARGETS
  umpire_tpl_judy
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

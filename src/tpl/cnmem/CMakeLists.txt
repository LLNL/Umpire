blt_add_library(
  NAME umpire_tpl_cnmem
  HEADERS cnmem.h
  SOURCES cnmem.cpp
  HEADERS_OUTPUT_SUBDIR umpire/tpl/cnmem
  DEPENDS_ON cuda_runtime)

target_compile_definitions(umpire_tpl_cnmem PUBLIC -DCNMEM_DLLEXPORT)

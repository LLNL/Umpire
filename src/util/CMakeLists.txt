set (umpire_util_headers
  AllocationMap.hpp
  AllocationRecord.hpp
  AllocatorTraits.hpp
  Exception.hpp
  Logger.hpp
  Macros.hpp
  Platform.hpp)

set (umpire_util_sources
  AllocationMap.cpp
  Exception.cpp
  Logger.cpp)

if (ENABLE_SLIC)
  set(slicDepend umpire_tpl_slic)
endif()

blt_add_library(
  NAME umpire_util
  HEADERS ${umpire_util_headers}
  SOURCES ${umpire_util_sources}
  HEADERS_OUTPUT_SUBDIR umpire/util
  DEPENDS_ON umpire_tpl_judy ${slicDepend}
)

install(TARGETS
  umpire_util
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)


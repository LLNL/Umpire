##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

set (umpire_event_headers
  event.hpp
  event_store.hpp
  event_store_recorder.hpp
  json_file_store.hpp
  recorder_chain.hpp
  recorder_factory.hpp)

set (umpire_event_sources
  event_store_recorder.cpp
  json_file_store.cpp
  recorder_chain.cpp
  recorder_factory.cpp)

set (umpire_event_depends ${UMPIRE_FMT_TARGET} umpire_tpl_json camp)

if (UMPIRE_ENABLE_CUDA)
  set(umpire_event_depends
    ${umpire_event_depends}
    cuda_runtime)
endif ()

if (UMPIRE_ENABLE_HIP)
  set(umpire_event_depends
    ${umpire_event_depends}
    blt::hip_runtime)
endif ()


if (UMPIRE_ENABLE_SQLITE_EXPERIMENTAL)
  set (umpire_event_headers ${umpire_event_headers} sqlite_database.hpp)
  set (umpire_event_sources ${umpire_event_sources} sqlite_database.cpp)
  set (umpire_event_depends ${umpire_event_depends} SQLite::SQLite3)
endif()

if (UMPIRE_ENABLE_STREAMING)
  set (umpire_event_headers ${umpire_event_headers}
    streaming_backend.hpp
    streaming_event_sink.hpp
    metrics_aggregator.hpp
    prometheus_backend.hpp)
  set (umpire_event_sources ${umpire_event_sources}
    streaming_event_sink.cpp
    metrics_aggregator.cpp
    prometheus_backend.cpp)

  # Find libcurl for HTTP POST to Prometheus
  find_package(CURL REQUIRED)

  # TODO: Add snappy for Prometheus remote-write protobuf compression
  # find_package(Snappy REQUIRED)
endif()

blt_add_library(
  NAME umpire_event
  HEADERS ${umpire_event_headers}
  SOURCES ${umpire_event_sources}
  DEPENDS_ON ${umpire_event_depends}
  OBJECT TRUE
)

target_include_directories(
  umpire_event
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if (UMPIRE_ENABLE_STREAMING)
  target_link_libraries(umpire_event PUBLIC ${CURL_LIBRARIES})
  target_include_directories(umpire_event SYSTEM PUBLIC ${CURL_INCLUDE_DIRS})
endif()

install(FILES
  ${umpire_event_headers}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/umpire/event)

umpire_add_code_checks(PREFIX event)

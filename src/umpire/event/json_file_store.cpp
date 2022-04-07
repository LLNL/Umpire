//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/event/json_file_store.hpp"

#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <string>
#include <vector>

#include "umpire/event/event_json.hpp"
#include "umpire/fmt/fmt.hpp"
#include "umpire/json/json.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace event {

json_file_store::json_file_store(const std::string& filename, bool read_only)
    : m_filename{filename}, m_read_only{read_only}
{
}

void json_file_store::insert(const event& e)
{
  open_store();
  nlohmann::json json_event = e;
  std::stringstream ss;
  ss << json_event;
  fprintf(m_fstream, "%s\n", ss.str().c_str());
}

void json_file_store::insert(const allocate& e)
{
  fprintf(m_fstream,
          R"({"category":"operation","name":"allocate")"
          R"(,"numeric_args":{"size":%ld})"
          R"(,"string_args":{"allocator_ref":"%p","pointer":"%p"})"
          R"(,"tags":{"replay":"true"})"
          R"(,"timestamp":%lld})"
          "\n",
          e.size, e.ref, e.ptr,
          static_cast<long long>(
              std::chrono::time_point_cast<std::chrono::nanoseconds>(e.timestamp).time_since_epoch().count()));
}

void json_file_store::insert(const named_allocate& e)
{
  fprintf(m_fstream,
          R"({"category":"operation","name":"named_allocate")"
          R"(,"numeric_args":{"size":%ld})"
          R"(,"string_args":{"allocator_ref":"%p","pointer":"%p","allocation_name":"%s"})"
          R"(,"tags":{"replay":"true"})"
          R"(,"timestamp":%lld})"
          "\n",
          e.size, e.ref, e.ptr, e.name.c_str(),
          static_cast<long long>(
              std::chrono::time_point_cast<std::chrono::nanoseconds>(e.timestamp).time_since_epoch().count()));
}

void json_file_store::insert(const deallocate& e)
{
  fprintf(m_fstream,
          R"({"category":"operation","name":"deallocate")"
          R"(,"string_args":{"allocator_ref":"%p","pointer":"%p"})"
          R"(,"tags":{"replay":"true"})"
          R"(,"timestamp":%lld})"
          "\n",
          e.ref, e.ptr,
          static_cast<long long>(
              std::chrono::time_point_cast<std::chrono::nanoseconds>(e.timestamp).time_since_epoch().count()));
}

std::vector<event> json_file_store::get_events()
#if !defined(_MSC_VER)
{
  char* line{NULL};
  size_t len{0};
  std::vector<event> events;
  std::size_t line_number{1};

  open_store();
  while (getline(&line, &len, m_fstream) != -1) {
    nlohmann::json json_event;
    event e;

    try {
      json_event = nlohmann::json::parse(line);
      e = json_event;
    } catch (...) {
      UMPIRE_ERROR(umpire::runtime_error,
                   umpire::fmt::format("json_file_store::get_events: Error parsing Line #{}", line_number));
    }

    events.push_back(e);

    line_number++;
  }

  free(line);
  return events;
}
#else
{
  std::string line;
  std::vector<event> events;
  std::size_t line_number{1};
  std::fstream f;

  f.open(m_filename, std::fstream::in);

  if (f.fail()) {
    UMPIRE_ERROR(umpire::runtime_error, umpire::fmt::format("Failed to open {}", m_filename));
  }

  while (std::getline(f, line)) {
    nlohmann::json json_event;
    event e;

    try {
      json_event = nlohmann::json::parse(line);
      e = json_event;
    } catch (...) {
      UMPIRE_ERROR(umpire::runtime_error,
                   umpire::fmt::format("json_file_store::get_events: Error parsing Line #{}", line_number));
    }

    events.push_back(e);

    line_number++;
  }

  return events;
}
#endif

void json_file_store::open_store()
{
  if (m_fstream == NULL) {
    m_fstream = fopen(m_filename.c_str(), m_read_only ? "r" : "w");

    if (m_fstream == NULL) {
      UMPIRE_ERROR(umpire::runtime_error, umpire::fmt::format("Failed to open {}", m_filename));
    }
  }
}

} // namespace event
} // namespace umpire

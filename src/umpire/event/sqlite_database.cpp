//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/event/sqlite_database.hpp"

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <sstream>
#include <string>

#include "umpire/config.hpp"
#include "umpire/event/event.hpp"
#include "umpire/event/event_json.hpp"
#include "umpire/json/json.hpp"

namespace umpire {
namespace event {

#undef UMPIRE_TRACE_SQEXEC
#if defined(UMPIRE_TRACE_SQEXEC)
#define UMPIRE_PRINT_SQEXE(sql) std::cout << "sqlite3_exec(\"" << sql << "\")" << std::endl
#else
#define UMPIRE_PRINT_SQEXE(sql)
#endif

#define UMP_SQ3_EXE(db, sql, proc, cb)                                                                           \
  {                                                                                                              \
    UMPIRE_PRINT_SQEXE(sql);                                                                                     \
    char* messageError;                                                                                          \
    int error = sqlite3_exec(db, sql, proc, cb, &messageError);                                                  \
    if (error) {                                                                                                 \
      std::cout << __PRETTY_FUNCTION__ << ":" << __LINE__ << " (" << error << ") " << messageError << std::endl; \
      exit(1);                                                                                                   \
    }                                                                                                            \
  }

sqlite_database::sqlite_database(const std::string& name)
{
  {
    int error = sqlite3_open(name.c_str(), &m_database);
    if (error) {
      std::cout << __PRETTY_FUNCTION__ << " " << __LINE__ << ": sqlite3_open(" << name << ") failed with " << error
                << ": " << sqlite3_errstr(error) << std::endl;
    }
  }
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS EVENTS ("
      "EVENT           JSON    );";
  UMP_SQ3_EXE(m_database, sql.c_str(), NULL, 0);
}

void sqlite_database::insert(const event& e)
{
  nlohmann::json json_event = e;
  const std::string sql{"INSERT INTO EVENTS VALUES(json('" + json_event.dump() + "'));"};
  UMP_SQ3_EXE(m_database, sql.c_str(), NULL, 0);
}

void sqlite_database::insert(const allocate& e)
{
  char buffer[512];
  sprintf(buffer,
          "INSERT INTO EVENTS VALUES(json('"
          R"({"category":"operation","name":"allocate")"
          R"(,"numeric_args":{"size":%ld})"
          R"(,"string_args":{"allocator_ref":"%p","pointer":"%p"})"
          R"(,"tags":{"replay":"true"})"
          R"(,"timestamp":%lld})"
          "'));",
          e.size, e.ref, e.ptr,
          static_cast<long long>(
              std::chrono::time_point_cast<std::chrono::nanoseconds>(e.timestamp).time_since_epoch().count()));

  UMP_SQ3_EXE(m_database, buffer, NULL, 0);
}

void sqlite_database::insert(const named_allocate& e)
{
  char buffer[512];

  sprintf(buffer,
          "INSERT INTO EVENTS VALUES(json('"
          R"({"category":"operation","name":"named_allocate")"
          R"(,"numeric_args":{"size":%ld})"
          R"(,"string_args":{"allocator_ref":"%p","pointer":"%p","allocation_name":"%s"})"
          R"(,"tags":{"replay":"true"})"
          R"(,"timestamp":%lld})"
          "'));",
          e.size, e.ref, e.ptr, e.name.c_str(),
          static_cast<long long>(
              std::chrono::time_point_cast<std::chrono::nanoseconds>(e.timestamp).time_since_epoch().count()));

  UMP_SQ3_EXE(m_database, buffer, NULL, 0);
}

void sqlite_database::insert(const allocate_resource& e)
{
  char buffer[512];

  sprintf(buffer,
          "INSERT INTO EVENTS VALUES(json('"
          R"({"category":"operation","name":"allocate_resource")"
          R"(,"numeric_args":{"size":%ld})"
          R"(,"string_args":{"allocator_ref":"%p","pointer":"%p","resource":"%s"})"
          R"(,"tags":{"replay":"true"})"
          R"(,"timestamp":%lld})"
          "'));",
          e.size, e.ref, e.ptr, e.res.c_str(),
          static_cast<long long>(
              std::chrono::time_point_cast<std::chrono::nanoseconds>(e.timestamp).time_since_epoch().count()));

  UMP_SQ3_EXE(m_database, buffer, NULL, 0);
}

void sqlite_database::insert(const deallocate& e)
{
  char buffer[512];

  sprintf(buffer,
          "INSERT INTO EVENTS VALUES(json('"
          R"({"category":"operation","name":"deallocate")"
          R"(,"string_args":{"allocator_ref":"%p","pointer":"%p"})"
          R"(,"tags":{"replay":"true"})"
          R"(,"timestamp":%lld})"
          "'));",
          e.ref, e.ptr,
          static_cast<long long>(
              std::chrono::time_point_cast<std::chrono::nanoseconds>(e.timestamp).time_since_epoch().count()));

  UMP_SQ3_EXE(m_database, buffer, NULL, 0);
}

void sqlite_database::insert(const deallocate_resource& e)
{
  char buffer[512];

  sprintf(buffer,
          "INSERT INTO EVENTS VALUES(json('"
          R"({"category":"operation","name":"deallocate")"
          R"(,"string_args":{"allocator_ref":"%p","pointer":"%p", "resource":"%s"})"
          R"(,"tags":{"replay":"true"})"
          R"(,"timestamp":%lld})"
          "'));",
          e.ref, e.ptr, e.res.c_str(),
          static_cast<long long>(
              std::chrono::time_point_cast<std::chrono::nanoseconds>(e.timestamp).time_since_epoch().count()));

  UMP_SQ3_EXE(m_database, buffer, NULL, 0);
}

std::vector<event> sqlite_database::get_events()
{
  std::vector<event> events;

  const auto event_processor = [](void* data, int, char** argv, char**) -> int {
    std::vector<event>* events = reinterpret_cast<std::vector<event>*>(data);

    auto j = nlohmann::json::parse(argv[0]);
    event e = j.get<event>();
    events->push_back(e);

    return 0;
  };

  const std::string sql{"SELECT * FROM EVENTS;"};
  UMP_SQ3_EXE(m_database, sql.c_str(), event_processor, (void*)&events);

  return events;
}

} // namespace event
} // namespace umpire

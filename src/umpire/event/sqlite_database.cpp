//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/event/sqlite_database.hpp"

#include <iostream>

#include "umpire/config.hpp"
#include "umpire/event/event.hpp"
#include "umpire/event/event_json.hpp"
#include "umpire/tpl/json/json.hpp"

namespace umpire {
namespace event {

sqlite_database::sqlite_database(const std::string& name)
{
  sqlite3_open(name.c_str(), &m_database);
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS EVENTS ("
      "EVENT           JSON    );";
  char* messaggeError;
  sqlite3_exec(m_database, sql.c_str(), NULL, 0, &messaggeError);
}

void sqlite_database::insert(event e)
{
  nlohmann::json json_event = e;
  const std::string sql{"INSERT INTO EVENTS VALUES(json('" + json_event.dump() + "'));"};
  sqlite3_exec(m_database, sql.c_str(), NULL, 0, NULL);
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
  sqlite3_exec(m_database, sql.c_str(), event_processor, (void*)&events, NULL);

  return events;
}

} // namespace event
} // namespace umpire

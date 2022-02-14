//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/event/json_file_store.hpp"

#include <fstream>
#include <string>
#include <vector>

#include "umpire/event/event_json.hpp"
#include "umpire/tpl/json/json.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace event {

json_file_store::json_file_store(const std::string& filename, bool read_only)
    : m_filename{filename}, m_read_only{read_only}
{
}

void json_file_store::insert(event e)
{
  open_store();
  nlohmann::json json_event = e;
  m_fstream << json_event << std::endl;
}

void json_file_store::insert_direct(const std::string& s)
{
  open_store();
  m_fstream << s << std::endl;
}

std::vector<event> json_file_store::get_events()
{
  std::string line;
  std::vector<event> events;
  std::size_t line_number{1};

  open_store();
  while (std::getline(m_fstream, line)) {
    nlohmann::json json_event;
    event e;

    try {
      json_event = nlohmann::json::parse(line);
      e = json_event;
    } catch (...) {
      UMPIRE_ERROR("json_file_store::get_events: Error parsing Line #" << line_number);
    }

    events.push_back(e);

    line_number++;
  }

  return events;
}

void json_file_store::open_store()
{
  if (!m_fstream.is_open()) {
    std::fstream::openmode mode{m_read_only ? std::fstream::in : std::fstream::out | std::fstream::trunc};

    m_fstream.open(m_filename, mode);

    if (m_fstream.fail()) {
      UMPIRE_ERROR("Failed to open " << m_filename);
    }
  }
}

} // namespace event
} // namespace umpire

//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/event/event_store_recorder.hpp"

#include <string>

#include "umpire/event/event.hpp"

namespace umpire {
namespace event {

event_store_recorder::event_store_recorder(event_store* db) : m_database(db)
{
}

void event_store_recorder::record(event e)
{
  m_database->insert(e);
}

void event_store_recorder::record_direct(const std::string& s)
{
  m_database->insert_direct(s);
}

} // namespace event
} // namespace umpire

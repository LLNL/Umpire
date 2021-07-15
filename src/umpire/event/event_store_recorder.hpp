//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_event_store_recorder_HPP
#define UMPIRE_event_store_recorder_HPP

#include "umpire/event/event_store.hpp"

namespace umpire {
namespace event {

class event;

class event_store_recorder {
 public:
  event_store_recorder(event_store* db);

  void record(event e);

 private:
  event_store* m_database;
};

} // namespace event
} // namespace umpire
#endif // UMPIRE_event_store_recorder_HPP

//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_sqlite_database_HPP
#define UMPIRE_sqlite_database_HPP

#include "umpire/config.hpp"
#ifdef UMPIRE_ENABLE_SQLITE_EXPERIMENTAL
#include <string>

#include "sqlite3.h"
#include "umpire/event/event_store.hpp"

namespace umpire {
namespace event {

struct event;
struct allocate;
struct named_allocate;
struct allocate_resource;
struct deallocate;
struct deallocate_resource;

class sqlite_database : public event_store {
 public:
  sqlite_database(const std::string& name);

  virtual void insert(const event& e) override final;
  virtual void insert(const allocate& e) override final;
  virtual void insert(const named_allocate& e) override final;
  virtual void insert(const allocate_resource& e) override final;
  virtual void insert(const deallocate& e) override final;
  virtual void insert(const deallocate_resource& e) override final;

  std::vector<event> get_events() override final;

 private:
  sqlite3* m_database;
};

} // namespace event
} // namespace umpire
#endif // UMPIRE_ENABLE_SQLITE_EXPERIMENTAL

#endif // UMPIRE_sqlite_database_HPP

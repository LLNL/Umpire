//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_quest_database_HPP
#define UMPIRE_quest_database_HPP

#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <string>

#include "umpire/event/event_store.hpp"

namespace umpire {
namespace event {

struct event;
struct allocate;
struct named_allocate;
struct allocate_resource;
struct deallocate;
struct deallocate_resource;

class quest_database : public event_store {
 public:
  quest_database(const std::string& url, const std::string& port, const std::string& name);

  void insert(const event& e) override final;
  void insert(const allocate& e) override final;
  void insert(const named_allocate& e) override final;
  void insert(const allocate_resource& e) override final;
  void insert(const deallocate& e) override final;
  void insert(const deallocate_resource& e) override final;

  std::vector<event> get_events() override final;

 private:
  const std::string m_url;
  const std::string m_port;
  const std::string m_name;

  int m_socket_desc;
  addrinfo* m_db_server;
};

} // namespace event
} // namespace umpire

#endif // UMPIRE_quest_database_HPP

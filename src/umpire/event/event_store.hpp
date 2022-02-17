//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_event_store_HPP
#define UMPIRE_event_store_HPP

#include <string>
#include <vector>

namespace umpire {
namespace event {

struct event;
struct allocate;
struct named_allocate;
struct deallocate;

class event_store {
 public:
  virtual void insert(const event& e) = 0;
  virtual void insert(const allocate& e) = 0;
  virtual void insert(const named_allocate& e) = 0;
  virtual void insert(const deallocate& e) = 0;

  virtual std::vector<event> get_events() = 0;
};

} // namespace event
} // namespace umpire

#endif // UMPIRE_event_store_HPP

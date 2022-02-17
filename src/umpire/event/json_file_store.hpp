//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_json_file_store_HPP
#define UMPIRE_json_file_store_HPP

#include <stdio.h>

#include <string>
#include <vector>

#include "umpire/event/event_store.hpp"

namespace umpire {
namespace event {

struct event;
struct allocate;
struct named_allocate;
struct deallocate;

class json_file_store : public event_store {
 public:
  json_file_store(const std::string& filename, bool read_only = false);

  virtual void insert(const event& e);
  virtual void insert(const allocate& e);
  virtual void insert(const named_allocate& e);
  virtual void insert(const deallocate& e);

  virtual std::vector<event> get_events();

 private:
  void open_store();
  FILE* m_fstream{nullptr};
  std::string m_filename;
  bool m_read_only;
};

} // namespace event
} // namespace umpire
#endif // UMPIRE_json_file_store_HPP

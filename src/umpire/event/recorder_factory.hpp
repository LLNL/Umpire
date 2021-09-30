//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_recorder_factory_HPP
#define UMPIRE_recorder_factory_HPP

#include "umpire/event/event_store_recorder.hpp"

namespace umpire {
namespace event {

using store_type = event_store_recorder;

class recorder_factory {
 public:
  static store_type& get_recorder();
};

} // namespace event
} // namespace umpire

#endif

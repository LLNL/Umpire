//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipAllocator_HPP
#define UMPIRE_HipAllocator_HPP

#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"
#include "umpire/strategy/GranularityController.hpp"

namespace umpire {
namespace alloc {

struct HipAllocator {
  umpire::strategy::GranularityController::Granularity
  set_granularity(umpire::strategy::GranularityController::Granularity gran) {
      umpire::strategy::GranularityController::Granularity old_granularity{ m_granularity };
      m_granularity = gran;
      return old_granularity;
  };

  umpire::strategy::GranularityController::Granularity
  m_granularity{umpire::strategy::GranularityController::Granularity::Default};
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_HipAllocator_HPP

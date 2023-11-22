//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipMemoryResource_HPP
#define UMPIRE_HipMemoryResource_HPP

#include "umpire/strategy/GranularityController.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace resource {

struct HipMemoryResource {
  virtual umpire::strategy::GranularityController::Granularity set_granularity(
      umpire::strategy::GranularityController::Granularity gran) = 0;
};

template <typename Derived>
struct HipMemoryResourceProxy : HipMemoryResource {
  virtual umpire::strategy::GranularityController::Granularity set_granularity(
      umpire::strategy::GranularityController::Granularity gran)
  {
    Derived* p = dynamic_cast<Derived*>(this);
    return p->m_allocator.set_granularity(gran);
  };
};

} // namespace resource
} // end of namespace umpire

#endif // UMPIRE_HipMemoryResource_HPP

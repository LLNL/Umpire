//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/memory.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief Base class to represent the available hardware resources for memory
 * allocation in the system.
 *
 * Objects of this inherit from strategy::AllocationStrategy, allowing them to
 * be used directly.
 */
template<typename Platform>
struct memory_resource :
  public memory
{
  using platform = Platform;

  memory_resource(const std::string& name) :
    memory(name) {}
};

} // end of namespace resource
} // end of namespace umpire

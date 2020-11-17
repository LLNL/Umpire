//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_AllocationStrategy_HPP
#define UMPIRE_AllocationStrategy_HPP

#include <cstddef>
#include <memory>
#include <ostream>
#include <string>

#include "umpire/Memory.hpp"

#include "umpire/util/MemoryResourceTraits.hpp"
#include "umpire/util/Platform.hpp"

namespace umpire {
namespace strategy {

/*!
 * \brief AllocationStrategy provides a unified interface to all classes that
 * can be used to allocate and free data.
 */
class AllocationStrategy :
  public Memory {
 public:
  /*!
   * \brief Construct a new AllocationStrategy object.
   *
   * All AllocationStrategy objects must will have a unique name and id. This
   * uniqueness is enforced by the ResourceManager.
   *
   * \param name The name of this AllocationStrategy object.
   * \param id The id of this AllocationStrategy object.
   */
  AllocationStrategy(const std::string& name, int id, Memory* parent) noexcept;

  virtual ~AllocationStrategy() = default;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_AllocationStrategy_HPP

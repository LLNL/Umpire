//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MemoryResource_HPP
#define UMPIRE_MemoryResource_HPP

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief Base class to represent the available hardware resources for memory
 * allocation in the system.
 *
 * Objects of this inherit from strategy::AllocationStrategy, allowing them to
 * be used directly.
 */
class MemoryResource : public strategy::AllocationStrategy {
 public:
  /*!
   * \brief Construct a MemoryResource with the given name and id.
   *
   * \param name Name of the MemoryResource.
   * \param id ID of the MemoryResource (must be unique).
   *
   */
  MemoryResource(const std::string& name, int id, MemoryResourceTraits traits);

  virtual ~MemoryResource() = default;

  /*!
   * \brief Allocate bytes of memory.
   *
   * This function is pure virtual and must be implemented by the inheriting
   * class.
   *
   * \param bytes Number of bytes to allocate.
   *
   * \return Pointer to start of allocation.
   */
  virtual void* allocate(std::size_t bytes) override = 0;

  /*!
   * \brief Free the memory at ptr.
   *
   * This function is pure virtual and must be implemented by the inheriting
   * class.
   *
   * \param ptr Pointer to free.
   */
  virtual void deallocate(void* ptr) override = 0;

  /*!
   * \brief Get the Platform assocatiated with this MemoryResource.
   *
   * This function is pure virtual and must be implemented by the inheriting
   * class.
   *
   * \return Platform associated with this MemoryResource.
   */
  virtual Platform getPlatform() noexcept override = 0;

  /*
   * \brief Check whether the Platform is accessible from a Memory
   * Resource
   *
   * \return true if the Platform p is accessible.
   */
  virtual bool isAccessibleFrom(Platform p) noexcept = 0;

  MemoryResourceTraits getTraits() const noexcept override;
 
 protected:
  MemoryResourceTraits m_traits;
};

} // namespace resource
} // end of namespace umpire

#endif // UMPIRE_MemoryResource_HPP

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
   * \brief Return the current size of this MemoryResource.
   *
   * This is sum of the sizes of all the tracked allocations. Note that this
   * doesn't ever have to be equal to getHighWatermark.
   *
   * \return current total size of active allocations in this MemoryResource.
   */
  virtual std::size_t getCurrentSize() const noexcept override = 0;

  /*!
   * \brief Return the memory high watermark for this MemoryResource.
   *
   * This is the largest amount of memory allocated by this Allocator. Note
   * that this may be larger than the largest value returned by
   * getCurrentSize.
   *
   * \return Memory high watermark.
   */
  virtual std::size_t getHighWatermark() const noexcept override = 0;

  /*!
   * \brief Get the Platform assocatiated with this MemoryResource.
   *
   * This function is pure virtual and must be implemented by the inheriting
   * class.
   *
   * \return Platform associated with this MemoryResource.
   */
  virtual Platform getPlatform() noexcept override = 0;

  MemoryResourceTraits getTraits() const noexcept override;

 protected:
  MemoryResourceTraits m_traits;
};

} // namespace resource
} // end of namespace umpire

#endif // UMPIRE_MemoryResource_HPP

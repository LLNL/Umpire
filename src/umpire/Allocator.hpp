//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Allocator_HPP
#define UMPIRE_Allocator_HPP

#include <cstddef>
#include <memory>
#include <ostream>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/Platform.hpp"

class AllocatorTest;

namespace umpire {

class ResourceManager;

/*!
 * \brief Provides a unified interface to allocate and free data.
 *
 * An Allocator encapsulates all the details of how and where allocations will
 * be made, and can also be used to introspect the memory resource. Allocator
 * objects do not return typed allocations, so the pointer returned from the
 * allocate method must be cast to the relevant type.
 *
 * \see TypedAllocator
 */
class Allocator {
  friend class ResourceManager;
  friend class ::AllocatorTest;

 public:
  /*!
   * \brief Allocate bytes of memory.
   *
   * The memory will be allocated as determined by the AllocationStrategy
   * used by this Allocator. Note that this method does not guarantee new
   * memory pages being requested from the underlying memory system, as the
   * associated AllocationStrategy could have already allocated sufficient
   * memory, or re-use existing allocations that were not returned to the
   * system.
   *
   * \param bytes Number of bytes to allocate (>= 0)
   *
   * \return Pointer to start of the allocation.
   */
  inline void* allocate(std::size_t bytes);

  /*!
   * \brief Free the memory at ptr.
   *
   * This method will throw an umpire::Exception if ptr was not allocated
   * using this Allocator. If you need to deallocate memory allocated by an
   * unknown object, use the ResourceManager::deallocate method.
   *
   * \param ptr Pointer to free (!nullptr)
   */
  inline void deallocate(void* ptr);

  /*!
   * \brief Release any and all unused memory held by this Allocator.
   */
  void release();

  /*!
   * \brief Return number of bytes allocated for allocation
   *
   * \param ptr Pointer to allocation in question
   *
   * \return number of bytes allocated for ptr
   */
  std::size_t getSize(void* ptr) const;

  /*!
   * \brief Return the memory high watermark for this Allocator.
   *
   * This is the largest amount of memory allocated by this Allocator. Note
   * that this may be larger than the largest value returned by
   * getCurrentSize.
   *
   * \return Memory high watermark.
   */
  std::size_t getHighWatermark() const noexcept;

  /*!
   * \brief Return the current size of this Allocator.
   *
   * This is sum of the sizes of all the tracked allocations. Note that this
   * doesn't ever have to be equal to getHighWatermark.
   *
   * \return current size of Allocator.
   */
  std::size_t getCurrentSize() const noexcept;

  /*!
   * \brief Return the actual size of this Allocator.
   *
   * For non-pool allocators, this will be the same as getCurrentSize().
   *
   * For pools, this is the total amount of memory allocated for blocks
   * managed by the pool.
   *
   * \return actual size of Allocator.
   */
  std::size_t getActualSize() const noexcept;

  /*!
   * \brief Return the number of active allocations.
   */
  std::size_t getAllocationCount() const noexcept;

  /*!
   * \brief Get the name of this Allocator.
   *
   * Allocators are uniquely named, and the name of the Allocator can be used
   * to retrieve the same Allocator from the ResourceManager at a later time.
   *
   * \see ResourceManager::getAllocator
   *
   * \return name of Allocator.
   */
  const std::string& getName() const noexcept;

  /*!
   * \brief Get the integer ID of this Allocator.
   *
   * Allocators are uniquely identified, and the ID of the Allocator can be
   * used to retrieve the same Allocator from the ResourceManager at a later
   * time.
   *
   * \see ResourceManager::getAllocator
   *
   * \return integer id of Allocator.
   */
  int getId() const noexcept;

  /*!
   * \brief Get the AllocationStrategy object used by this Allocator.
   *
   *
   *
   * \return Pointer to the AllocationStrategy.
   */
  strategy::AllocationStrategy* getAllocationStrategy() noexcept;

  /*!
   * \brief Get the Platform object appropriate for this Allocator.
   *
   * \return Platform for this Allocator.
   */
  Platform getPlatform() noexcept;

  Allocator() = default;

  friend std::ostream& operator<<(std::ostream&, const Allocator&);

 private:
  /*!
   * \brief Construct an Allocator with the given AllocationStrategy.
   *
   * This method is private to ensure that only the ResourceManager can
   * construct Allocators.
   *
   * \param allocator Pointer to the AllocationStrategy object to use for
   * Allocations.
   */
  Allocator(strategy::AllocationStrategy* allocator) noexcept;

  /*!
   * \brief Pointer to the AllocationStrategy used by this Allocator.
   */
  umpire::strategy::AllocationStrategy* m_allocator;
};

} // end of namespace umpire

#include "umpire/Allocator.inl"

#endif // UMPIRE_Allocator_HPP

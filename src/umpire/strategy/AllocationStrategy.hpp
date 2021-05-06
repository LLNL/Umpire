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

#include "umpire/util/MemoryResourceTraits.hpp"
#include "umpire/util/Platform.hpp"

namespace umpire {

class ResourceManager;
class Allocator;

namespace strategy {

/*!
 * \brief AllocationStrategy provides a unified interface to all classes that
 * can be used to allocate and free data.
 */
class AllocationStrategy
{
  friend class umpire::ResourceManager;
  friend class umpire::Allocator;
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
  AllocationStrategy(const std::string& name, int id, AllocationStrategy* parent, const std::string& strategy_name) noexcept;

  virtual ~AllocationStrategy() = default;

  void* allocate_internal(std::size_t bytes);

  void deallocate_internal(void* ptr, std::size_t size=0);

  /*!
   * \brief Release any and all unused memory held by this AllocationStrategy
   */
  virtual void release();

  /*!
   * \brief Get current (total) size of the allocated memory.
   *
   * This is the total size of all allocation currently 'live' that have been
   * made by this AllocationStrategy object.
   *
   * \return Current total size of allocations.
   */
  virtual std::size_t getCurrentSize() const noexcept;

  /*!
   * \brief Get the high watermark of the total allocated size.
   *
   * This is equivalent to the highest observed value of getCurrentSize.
   * \return High watermark allocation size.
   */
  virtual std::size_t getHighWatermark() const noexcept;

  /*!
   * \brief Get the current amount of memory allocated by this allocator.
   *
   * Note that this can be larger than getCurrentSize(), particularly if the
   * AllocationStrategy implements some kind of pooling.
   *
   * \return The total size of all the memory this object has allocated.
   */
  virtual std::size_t getActualSize() const noexcept;

  /*!
   * \brief Get the total number of active allocations by this allocator.
   *
   * \return The total number of active allocations this object has allocated.
   */
  virtual std::size_t getAllocationCount() const noexcept;

  /*!
   * \brief Get the platform associated with this AllocationStrategy.
   *
   * The Platform distinguishes the appropriate place to execute operations
   * on memory allocated by this AllocationStrategy.
   *
   * \return The platform associated with this AllocationStrategy.
   */
  virtual Platform getPlatform() noexcept = 0;

  /*!
   * \brief Get the name of this AllocationStrategy.
   *
   * \return The name of this AllocationStrategy.
   */
  const std::string& getName() noexcept;

  const std::string& getStrategyName() const noexcept;

  /*!
   * \brief Get the id of this AllocationStrategy.
   *
   * \return The id of this AllocationStrategy.
   */
  int getId() noexcept;

  friend std::ostream& operator<<(std::ostream& os,
                                  const AllocationStrategy& strategy);

  /*!
   * \brief Traces where the allocator came from.
   *
   * \return Pointer to the parent AllocationStrategy.
   */
  virtual AllocationStrategy* getParent() const noexcept;

  virtual MemoryResourceTraits getTraits() const noexcept;

  virtual bool tracksMemoryUse() const noexcept;

  bool isTracked() const noexcept;

  std::size_t m_current_size{0};
  std::size_t m_high_watermark{0};
  std::size_t m_allocation_count{0};

 protected:
  void setTracking(bool) noexcept;

  std::string m_name;
  std::string m_strategy_name;
  int m_id;
  bool m_tracked{true};

  AllocationStrategy* m_parent; 
  private:

  /*!
   * \brief Allocate bytes of memory.
   *
   * \param bytes Number of bytes to allocate.
   *
   * \return Pointer to start of allocated bytes.
   */
  virtual void* allocate(std::size_t bytes) = 0;
  virtual void* allocate_named(const std::string& name, std::size_t bytes);

  /*!
   * \brief Free the memory at ptr.
   *
   * \param ptr Pointer to free.
   */
  virtual void deallocate(void* ptr, std::size_t size=0) = 0;

};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_AllocationStrategy_HPP

//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_AllocationStrategy_HPP
#define UMPIRE_AllocationStrategy_HPP

#include "umpire/util/Platform.hpp"

#include <string>
#include <memory>
#include <cstddef>
#include <ostream>

namespace umpire {
namespace strategy {

/*!
 * \brief AllocationStrategy provides a unified interface to all classes that
 * can be used to allocate and free data.
 */
class AllocationStrategy 
{
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
    AllocationStrategy(const std::string& name, int id) noexcept;

    virtual ~AllocationStrategy() = default;

    /*!
     * \brief Finalize an AllocationStrategy before destroying it.
     *
     *  After calling finalize only calls to deallocate are valid. 
     */
    virtual void finalize() = 0;

    /*!
     * \brief Allocate bytes of memory.
     *
     * \param bytes Number of bytes to allocate.
     *
     * \return Pointer to start of allocated bytes.
     */
    virtual void* allocate(size_t bytes) = 0;

    /*!
     * \brief Free the memory at ptr.
     *
     * \param ptr Pointer to free.
     */
    virtual void deallocate(void* ptr) = 0;

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
    virtual long getCurrentSize() const noexcept = 0;

    /*!
     * \brief Get the high watermark of the total allocated size.
     *
     * This is equivalent to the highest observed value of getCurrentSize.
     * \return High watermark allocation size.
     */
    virtual long getHighWatermark() const noexcept = 0;

    /*!
     * \brief Get the current amount of memory allocated by this allocator.
     *
     * Note that this can be larger than getCurrentSize(), particularly if the
     * AllocationStrategy implements some kind of pooling.
     *
     * \return The total size of all the memory this object has allocated.
     */
    virtual long getActualSize() const noexcept;

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


    /*!
     * \brief Get the id of this AllocationStrategy.
     *
     * \return The id of this AllocationStrategy.
     */
    int getId() noexcept;

    friend std::ostream& operator<<(std::ostream& os, const AllocationStrategy& strategy);

  protected:
    std::string m_name;

    int m_id;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_AllocationStrategy_HPP

#ifndef UMPIRE_MemoryResource_HPP
#define UMPIRE_MemoryResource_HPP

#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief Allocator provides a unified interface to all Umpire classes that can
 * be used to allocate and free data.
 */
class MemoryResource :
  public strategy::AllocationStrategy
{
  public:
    /*!
     * \brief Allocate bytes of memory.
     *
     * \param bytes Number of bytes to allocate.
     *
     * \return Pointer to start of allocation.
     */
    virtual void* allocate(size_t bytes) = 0;

    /*!
     * \brief Free the memory at ptr.
     *
     * \param ptr Pointer to free.
     */
    virtual void deallocate(void* ptr) = 0;

    /*!
     * \brief Return number of bytes allocated for allocation
     *
     * \param ptr Pointer to allocation in question
     *
     * \return number of bytes allocated for ptr
     */
    virtual size_t getSize(void* ptr) = 0;

    virtual long getCurrentSize() = 0;
    virtual long getHighWatermark() = 0;

    virtual Platform getPlatform()  = 0;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_MemoryResource_HPP

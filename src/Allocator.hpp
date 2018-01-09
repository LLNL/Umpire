#ifndef UMPIRE_Allocator_HPP
#define UMPIRE_Allocator_HPP

#include "umpire/strategy/AllocationStrategy.hpp"

#include <memory>
#include <cstddef>

namespace umpire {

class ResourceManager;

/*!
 * \brief Allocator provides a unified interface to all Umpire classes that can
 * be used to allocate and free data.
 */
class Allocator {
  friend class ResourceManager;

  public:
    /*!
     * \brief Allocate bytes of memory.
     *
     * \param bytes Number of bytes to allocate.
     *
     * \return Pointer to start of allocation.
     */
    void* allocate(size_t bytes);

    /*!
     * \brief Free the memory at ptr.
     *
     * \param ptr Pointer to free.
     */
    void deallocate(void* ptr);

    /*!
     * \brief Return number of bytes allocated for allocation
     *
     * \param ptr Pointer to allocation in question
     *
     * \return number of bytes allocated for ptr
     */
    size_t getSize(void* ptr);

    /*!
     * \brief Return the memory high watermark for this Allocator.
     *
     * This is the largest seen value of getCurrentSize.
     *
     * \return memory high watermark.
     */
    size_t getHighWatermark();

    /*!
     * \brief Return the current size of this Allocator.
     *
     * This is sum of the sizes of all the tracked allocations.
     *
     * \return current size of Allocator.
     */
    size_t getCurrentSize();

    /*!
     * \brief Get the name of this Allocator.
     *
     * \return name of Allocator.
     */
    std::string getName();

    std::shared_ptr<umpire::strategy::AllocationStrategy> getAllocationStrategy();

  private:
    Allocator(std::shared_ptr<strategy::AllocationStrategy>& allocator);

    std::shared_ptr<umpire::strategy::AllocationStrategy> m_allocator;
};

} // end of namespace umpire

#endif // UMPIRE_Allocator_HPP

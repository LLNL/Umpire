#ifndef UMPIRE_Allocator_HPP
#define UMPIRE_Allocator_HPP

#include "umpire/AllocatorInterface.hpp"

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
     * \return Number of bytes allocated
     */
    size_t size(void* ptr);

  private:
    Allocator(std::shared_ptr<umpire::AllocatorInterface>& allocator);
    std::shared_ptr<umpire::AllocatorInterface> m_allocator;
};

} // end of namespace umpire

#endif // UMPIRE_Allocator_HPP

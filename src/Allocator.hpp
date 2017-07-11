#ifndef UMPIRE_Allocator_HPP
#define UMPIRE_Allocator_HPP

#include <cstddef>

namespace umpire {

/*!
 * \brief Allocator provides a unified interface to all Umpire classes that can
 * be used to allocate and free data.
 */
class Allocator {
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
    virtual void free(void* ptr) = 0;
};

} // end of namespace umpire

#endif // UMPIRE_Allocator_HPP

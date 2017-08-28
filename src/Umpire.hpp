#ifndef UMPIRE_Umpire_HPP
#define UMPIRE_Umpire_HPP

#include "umpire/ResourceManager.hpp"

namespace umpire {

/*!
 * \brief Allocate memory in the default space, with the default allocator.
 *
 * This method is a convenience wrapper around calls to the ResourceManager to
 * allocate memory in the default MemorySpace.
 *
 * \param size Number of bytes to allocate.
 */
inline
void* malloc(size_t size)
{
  //return ResourceManager::getInstance().allocate(size);
}

/*!
 * \brief Free any memory allocated with Umpire.
 *
 * This method is a convenience wrapper around calls to the ResourceManager, it
 * can be used to free allocations from any MemorySpace. *
 *
 * \param ptr Address to free.
 */
inline
void free(void* ptr)
{
  //return ResourceManager::getInstance().deallocate(ptr);
}

} // end of namespace umpire

#endif // UMPIRE_Umpire_HPP

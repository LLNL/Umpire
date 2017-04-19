
#include "umpire/ResourceManager.hpp"

namespace umpire {

enum Spaces {
  HOST,
  GPU,
  /* NUM_SPACES is last */
  NUM_SPACES
};

/**
 * @brief Allocate memory in the default space, with the default allocator.
 *
 * This method is a convenience wrapper around calls to the ResourceManager to
 * allocate memory in the default MemorySpace.
 *
 * @param size Number of bytes to allocate.
 */
inline
void* malloc(size_t size)
{
  return ResourceManager::getInstance().allocate(size);
}

/**
 * @brief Free any memory allocated with Umpire.
 *
 * This method is a convenience wrapper around calls to the ResourceManager, it
 * can be used to free allocations from any MemorySpace. *
 *
 * @param ptr Address to free.
 */
inline
void free(void* ptr)
{
  return ResourceManager::getInstance().free(ptr);
}

}

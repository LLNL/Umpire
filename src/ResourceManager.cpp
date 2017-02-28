/**
 * Project Untitled
 */


#include "ResourceManager.h"

/**
 * ResourceManager implementation
 */


/**
 * @return ResourceManager
 */
ResourceManager ResourceManager::getResourceManager() {
    return null;
}

void ResourceManager::getAvailableSpaces() {

}

/**
 * @param bytes
 * @return void*
 */
void* ResourceManager::allocate(size_t bytes) {
    return null;
}

/**
 * @param bytes
 * @param space
 * @return void*
 */
void* ResourceManager::allocate(size_t bytes, string space) {
    return null;
}

/**
 * @param bytes
 * @param space
 * @return void*
 */
void* ResourceManager::allocate(size_t bytes, MemorySpace space) {
    return null;
}

/**
 * @param space
 */
void ResourceManager::setDefaultSpace(MemorySpace space) {

}

/**
 * @return MemorySpace
 */
MemorySpace ResourceManager::getDefaultSpace() {
    return null;
}

/**
 * @param pointer
 * @param destination
 */
void ResourceManager::move(void* pointer, MemorySpace destination) {

}
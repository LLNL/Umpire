/**
 * Project Untitled
 */


#ifndef _RESOURCEMANAGER_H
#define _RESOURCEMANAGER_H

#include "Memory Space.h"


class ResourceManager {
public: 
    vector m_spaces;
    map m_allocations_to_spaces;
    MemorySpace m_default_space;
    Memory Space has many;
    
    ResourceManager getResourceManager();
    
    void getAvailableSpaces();
    
    /**
     * @param bytes
     */
    void* allocate(size_t bytes);
    
    /**
     * @param bytes
     * @param space
     */
    void* allocate(size_t bytes, string space);
    
    /**
     * @param bytes
     * @param space
     */
    void* allocate(size_t bytes, MemorySpace space);
    
    /**
     * @param space
     */
    void setDefaultSpace(MemorySpace space);
    
    MemorySpace getDefaultSpace();
    
    /**
     * @param pointer
     * @param destination
     */
    void move(void* pointer, MemorySpace destination);
};

#endif //_RESOURCEMANAGER_H
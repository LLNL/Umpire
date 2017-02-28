/**
 * Project Untitled
 */


#ifndef _MEMORY SPACE_H
#define _MEMORY SPACE_H

class Memory Space {
public: 
    string m_descriptor;
    map m_allocations;
    vector m_allocators;
    MemoryAllocator m_default_allocator;
    
    void getTotalSize();
    
    void getProperties();
    
    void getRemainingSize();
    
    /**
     * @param bytes
     */
    void* alloc(size_t bytes);
    
    /**
     * @param ptr
     */
    void free(void* ptr);
    
    void getDescriptor();
    
    /**
     * @param allocator
     */
    void setDefaultAllocator(MemoryAllocator allocator);
    
    MemoryAllocator getDefaultAllocator();
    
    vector getAllocators();
};

#endif //_MEMORY SPACE_H
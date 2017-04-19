#ifndef UMPIRE_ResourceManager_HPP
#define UMPIRE_ResourceManager_HPP

#include "umpire/MemorySpace.hpp"

namespace umpire {

class ResourceManager {
  public: 
    static ResourceManager& getInstance();
    
    std::vector<MemorySpace*> getAvailableSpaces();
    
    void* allocate(size_t bytes);
    void* allocate(size_t bytes, MemorySpace* space);

    void free(void* pointer);
    
    void setDefaultSpace(MemorySpace* space);
    MemorySpace& getDefaultSpace();
    
    /**
     * @brief
     *
     * @param pointer
     * @param destination
     */
    void move(void* pointer, MemorySpace& destination);

  protected:
    ResourceManager();

  private:
    static ResourceManager* s_resource_manager_instance;

    std::vector<MemorySpace*> m_spaces;
    std::map<void*, MemorySpace*> m_allocation_spaces;
    MemorySpace* m_default_space;

    long m_allocated;
};

} // end of namespace umpire

#endif // UMPIRE_ResourceManager_HPP

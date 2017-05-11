#ifndef UMPIRE_ResourceManager_HPP
#define UMPIRE_ResourceManager_HPP

#include "umpire/Allocator.hpp"
#include "umpire/space/MemorySpace.hpp"

namespace umpire {

class ResourceManager : public Allocator {
  public: 
    static ResourceManager& getInstance();
    
    std::vector<std::string> getAvailableSpaces();

    space::MemorySpace* findSpace();
    
    /*
     */
    virtual void* allocate(size_t bytes);
    virtual void free(void* pointer);

    void* allocate(size_t bytes, space::MemorySpace* space);

    
    void setDefaultSpace(space::MemorySpace* space);
    space::MemorySpace& getDefaultSpace();
    
    /**
     * @brief
     *
     * @param pointer
     * @param destination
     */
    void move(void* pointer, space::MemorySpace& destination);

  protected:
    ResourceManager();

  private:
    static ResourceManager* s_resource_manager_instance;

    std::map<std::string, space::MemorySpace*> m_spaces;
    std::map<void*, space::MemorySpace*> m_allocation_spaces;
    space::MemorySpace* m_default_space;

    long m_allocated;
};

} // end of namespace umpire

#endif // UMPIRE_ResourceManager_HPP

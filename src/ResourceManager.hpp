#ifndef UMPIRE_ResourceManager_HPP
#define UMPIRE_ResourceManager_HPP

#include "umpire/Allocator.hpp"
#include "umpire/space/MemorySpace.hpp"

#include <vector>
#include <string>
#include <memory>

namespace umpire {

class ResourceManager : public Allocator {
  public: 
    static ResourceManager& getInstance();
    
    std::vector<std::string> getAvailableSpaces();

    std::shared_ptr<space::MemorySpace> getSpace(const std::string& space);
    
    virtual void* allocate(size_t bytes);
    virtual void free(void* pointer);

    void* allocate(size_t bytes, std::shared_ptr<space::MemorySpace> space);

    void registerAllocation(void* ptr, std::shared_ptr<space::MemorySpace> space);

    
    void setDefaultSpace(std::shared_ptr<space::MemorySpace> space);
    std::shared_ptr<space::MemorySpace> getDefaultSpace();
    
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

    std::map<std::string, std::shared_ptr<space::MemorySpace>> m_spaces;
    std::map<void*, std::shared_ptr<space::MemorySpace>> m_allocation_spaces;
    std::shared_ptr<space::MemorySpace> m_default_space;

    long m_allocated;
};

} // end of namespace umpire

#endif // UMPIRE_ResourceManager_HPP

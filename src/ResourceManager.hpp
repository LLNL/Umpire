#ifndef UMPIRE_ResourceManager_HPP
#define UMPIRE_ResourceManager_HPP

#include "umpire/Allocator.hpp"

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace umpire {

class ResourceManager
{
  public: 
    static ResourceManager& getInstance();
    
    std::vector<std::string> getAvailableAllocators();

    std::shared_ptr<Allocator> getAllocator(const std::string& space);

    void setDefaultAllocator(std::shared_ptr<Allocator> space);
    std::shared_ptr<Allocator> getDefaultAllocator();
    
    void registerAllocation(void* ptr, std::shared_ptr<Allocator> space);
    void deregisterAllocation(void* ptr);
    
  protected:
    ResourceManager();

  private:
    static ResourceManager* s_resource_manager_instance;

    std::unordered_map<std::string, std::shared_ptr<Allocator>> m_allocators;
    std::unordered_map<void*, std::shared_ptr<Allocator>> m_allocation_to_allocator;
    std::shared_ptr<Allocator> m_default_space;

    long m_allocated;
};

} // end of namespace umpire

#endif // UMPIRE_ResourceManager_HPP

#ifndef UMPIRE_ResourceManager_HPP
#define UMPIRE_ResourceManager_HPP

#include "umpire/Allocator.hpp"
#include "umpire/AllocatorInterface.hpp"

#include <vector>
#include <string>
#include <memory>
#include <list>
#include <unordered_map>

namespace umpire {

class ResourceManager
{
  public: 
    static ResourceManager& getInstance();
    
    std::vector<std::string> getAvailableAllocators();

    Allocator getAllocator(const std::string& space);
    Allocator getAllocator(void* ptr);

    void setDefaultAllocator(Allocator allocator);
    Allocator getDefaultAllocator();
    
    void registerAllocation(void* ptr, std::shared_ptr<AllocatorInterface> space);
    void deregisterAllocation(void* ptr);

    void copy(void* src_ptr, void* dst_ptr);

    /*
     * \brief Deallocate any pointer allocated by an Umpire-managed resource.
     *
     * \param ptr Pointer to deallocate
     */
    void deallocate(void* ptr);
    
  private:
    ResourceManager();
    ResourceManager (const ResourceManager&) = delete;
    ResourceManager& operator= (const ResourceManager&) = delete;

    std::shared_ptr<AllocatorInterface>& findAllocatorForPointer(void* ptr);

    static ResourceManager* s_resource_manager_instance;

    std::list<std::string> m_allocator_names;

    std::unordered_map<std::string, std::shared_ptr<AllocatorInterface> > m_allocators;
    std::unordered_map<void*, std::shared_ptr<AllocatorInterface> > m_allocation_to_allocator;
    std::shared_ptr<AllocatorInterface> m_default_allocator;

    long m_allocated;
};

} // end of namespace umpire

#endif // UMPIRE_ResourceManager_HPP

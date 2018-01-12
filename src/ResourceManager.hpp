#ifndef UMPIRE_ResourceManager_HPP
#define UMPIRE_ResourceManager_HPP

#include <vector>
#include <string>
#include <memory>
#include <list>
#include <unordered_map>

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/AllocatorTraits.hpp"
#include "umpire/util/AllocationMap.hpp"

namespace umpire {

class ResourceManager
{
  public: 
    static ResourceManager& getInstance();

    /*!
     * \brief Initialize available memory systems.
     */
    void initialize();

    void finalize();
    
    std::vector<std::string> getAvailableAllocators();

    Allocator getAllocator(const std::string& space);

    Allocator makeAllocator(const std::string& name, 
        const std::string& strategy, 
        util::AllocatorTraits traits,
        std::vector<Allocator> providers);

    Allocator getAllocator(void* ptr);

    void setDefaultAllocator(Allocator allocator);
    Allocator getDefaultAllocator();
    
    void registerAllocation(void* ptr, util::AllocationRecord* record);
    void deregisterAllocation(void* ptr);

    void copy(void* src_ptr, void* dst_ptr, size_t size=0);

    void memset(void* ptr, int val, size_t length=0);

    /*
     * \brief Deallocate any pointer allocated by an Umpire-managed resource.
     *
     * \param ptr Pointer to deallocate
     */
    void deallocate(void* ptr);

    size_t getSize(void* ptr);

  private:
    ResourceManager();

    ResourceManager (const ResourceManager&) = delete;
    ResourceManager& operator= (const ResourceManager&) = delete;

    std::shared_ptr<strategy::AllocationStrategy>& findAllocatorForPointer(void* ptr);
    std::shared_ptr<strategy::AllocationStrategy>& getAllocationStrategy(const std::string& name);

    static ResourceManager* s_resource_manager_instance;

    std::list<std::string> m_allocator_names;

    std::unordered_map<std::string, std::shared_ptr<strategy::AllocationStrategy> > m_allocators;

    util::AllocationMap m_allocations;

    std::shared_ptr<strategy::AllocationStrategy> m_default_allocator;

    std::unordered_map<std::string, std::shared_ptr<strategy::AllocationStrategy> > m_memory_resources;

    long m_allocated;
};

} // end of namespace umpire

#endif // UMPIRE_ResourceManager_HPP

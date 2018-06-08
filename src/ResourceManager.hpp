//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_ResourceManager_HPP
#define UMPIRE_ResourceManager_HPP

#include <vector>
#include <string>
#include <memory>
#include <list>
#include <unordered_map>

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/AllocationMap.hpp"

#include "umpire/resource/MemoryResourceTypes.hpp"

namespace umpire {

/*!
 * \brief 
 */
class ResourceManager {
  public: 
    /*!
     * \brief 
     */
    static ResourceManager& getInstance();

    /*!
     * \brief Initialize the ResourceManager.
     *
     * This will create all registered MemoryResource objects
     */
    void initialize();

    void finalize();
    
    /*!
     * \brief Get the names of all available Allocator objects.
     */
    std::vector<std::string> getAvailableAllocators();

    /*!
     * \brief Get the Allocator with the given name.
     */
    Allocator getAllocator(const std::string& name);

    /*!
     * \brief Get the default Allocator for the given resource_type.
     */
    Allocator getAllocator(resource::MemoryResourceType resource_type);

    /*!
     * \brief Construct a new Allocator.
     *
     */
    template <typename Strategy,
             typename... Args>
    Allocator makeAllocator(
        const std::string& name, 
        Args&&... args);

    /*!
     * \brief Get the Allocator used to allocate ptr.
     *
     *
     *
     * \param ptr Pointer to find the Allocator for.
     * \return Allocator for the given ptr.
     */
    Allocator getAllocator(void* ptr);

    /*!
     * \brief Does the given pointer have an associated Allocator.
     *
     * \return True if the pointer has an associated Allocator.
     */
    bool hasAllocator(void* ptr);
    
    void registerAllocation(void* ptr, util::AllocationRecord* record);

    util::AllocationRecord* deregisterAllocation(void* ptr);

    /*!
     * \brief Check whether the named Allocator exists.
     *
     */
    bool isAllocatorRegistered(const std::string& name);

    /*!
     * \brief Copy size bytes of data from src_ptr to dst_ptr.
     *
     * Both the src_ptr and dst_ptr addresses must be allocated by Umpire. They
     * can be offset from any Umpire-managed base address.  
     *
     * The dst_ptr must be large enough to accommodate size bytes of data.
     *
     * \param dst_ptr Destination pointer.
     * \param src_ptr Source pointer.
     * \param size Size in bytes.
     */
    void copy(void* dst_ptr, void* src_ptr, size_t size=0);

    void transfer(void* dst_ptr, void* src_ptr, size_t size, std::shared_ptr<umpire::strategy::AllocationStrategy>& dst_alloc_strategy, std::shared_ptr<umpire::strategy::AllocationStrategy>& src_alloc_strategy);

    /*!
     * \brief Set the first length bytes of ptr to the value val.
     *
     * \param ptr Pointer to data.
     * \param val Value to set.
     * \param length Number of bytes to set to val.
     */
    void memset(void* ptr, int val, size_t length=0);

    /*!
     * \brief Reallocate src_ptr to size.
     *
     * \param src_ptr Source pointer to reallocate.
     * \param size New size of pointer.
     *
     * \return Reallocated pointer.
     *
     */
    void* reallocate(void* src_ptr, size_t size);

    /*!
     * \brief Move src_ptr to memory from allocator
     *
     * \param src_ptr Pointer to move.
     * \param allocator Allocator to use to allocate new memory for moved data.
     *
     * \return Pointer to new location of data.
     */
    void* move(void* src_ptr, Allocator allocator);

    /*!
     * \brief Deallocate any pointer allocated by an Umpire-managed resource.
     *
     * \param ptr Pointer to deallocate.
     */
    void deallocate(void* ptr);

    /*!
     * \brief Get the size in bytes of the allocation for the given pointer.
     *
     * \param ptr Pointer to find size of.
     *
     * \return Size of allocation in bytes.
     */
    size_t getSize(void* ptr);


  private:
    ResourceManager();

    ResourceManager (const ResourceManager&) = delete;
    ResourceManager& operator= (const ResourceManager&) = delete;

    std::shared_ptr<strategy::AllocationStrategy>& findAllocatorForPointer(void* ptr);
    std::shared_ptr<strategy::AllocationStrategy>& getAllocationStrategy(const std::string& name);

    int getNextId();

    static ResourceManager* s_resource_manager_instance;

    std::list<std::string> m_allocator_names;

    std::unordered_map<std::string, std::shared_ptr<strategy::AllocationStrategy> > m_allocators_by_name;
    std::unordered_map<int, std::shared_ptr<strategy::AllocationStrategy> > m_allocators_by_id;

    util::AllocationMap m_allocations;

    std::shared_ptr<strategy::AllocationStrategy> m_default_allocator;

    std::unordered_map<resource::MemoryResourceType, std::shared_ptr<strategy::AllocationStrategy>, resource::MemoryResourceTypeHash > m_memory_resources;

    long m_allocated;

    int m_id;
};

} // end of namespace umpire

#include "umpire/ResourceManager.inl"

#endif // UMPIRE_ResourceManager_HPP

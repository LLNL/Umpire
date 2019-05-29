//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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
#include <mutex>
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

    /*!
     * \brief Get the names of all available Allocator objects.
     */
    std::vector<std::string> getAllocatorNames() const noexcept;

    /*!
     * \brief Get the ids of all available Allocator objects.
     */
    std::vector<int> getAllocatorIds() const noexcept;

    /*!
     * \brief Get the Allocator with the given name.
     */
    Allocator getAllocator(const std::string& name);

    Allocator getAllocator(const char* name);

    /*!
     * \brief Get the default Allocator for the given resource_type.
     */
    Allocator getAllocator(resource::MemoryResourceType resource_type);

    /*!
     * \brief Get the Allocator with the given ID.
     */
    Allocator getAllocator(int id);

    /*!
     * \brief Get the default Allocator.
     *
     * The default Allocator is used whenever an Allocator is required and one
     * is not provided, or cannot be inferred.
     *
     * \return The default Allocator.
     */
    Allocator getDefaultAllocator();

    /*!
     * \brief Set the default Allocator.
     *
     * The default Allocator is used whenever an Allocator is required and one
     * is not provided, or cannot be inferred.
     *
     * \param allocator The Allocator to use as the default.
     */
    void setDefaultAllocator(Allocator allocator) noexcept;

    /*!
     * \brief Construct a new Allocator.
     */
    template <typename Strategy,
             bool introspection=true,
             typename... Args>
    Allocator makeAllocator(const std::string& name, Args&&... args);

    /*!
     * \brief Register an Allocator with the ResourceManager.
     *
     * After registration, the Allocator can be retrieved by calling
     * getAllocator(name).
     *
     * The same Allocator can be registered under multiple names.
     *
     * \param name Name to register Allocator with.
     * \param allocator Allocator to register.
     */
    void registerAllocator(const std::string& name, Allocator allocator);

    /*!
     * \brief Get the Allocator used to allocate ptr.
     *
     * \param ptr Pointer to find the Allocator for.
     * \return Allocator for the given ptr.
     */
    Allocator getAllocator(void* ptr);

    bool isAllocator(const std::string& name) noexcept;

    /*!
     * \brief Does the given pointer have an associated Allocator.
     *
     * \return True if the pointer has an associated Allocator.
     */
    bool hasAllocator(void* ptr);

    /*!
     * \brief register an allocation with the manager.
     */
    void registerAllocation(void* ptr, util::AllocationRecord record);

    /*!
     * \brief de-register the address ptr with the manager.
     *
     * \return the allocation record removed from the manager.
     */
    util::AllocationRecord deregisterAllocation(void* ptr);

    /*!
     * \brief Find the allocation record associated with an address ptr.
     *
     * \return the record if found, or throws an exception if not found.
     */
    const util::AllocationRecord* findAllocationRecord(void* ptr) const;

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
     * If src_ptr is null, then the default allocator will be used to allocate
     * data.
     *
     * \param src_ptr Source pointer to reallocate.
     * \param size New size of pointer.
     *
     * \return Reallocated pointer.
     *
     */
    void* reallocate(void* src_ptr, size_t size);

    /*!
     * \brief Reallocate src_ptr to size.
     *
     * If src_ptr is null, then allocator will be used to allocate the data.
     *
     * \param src_ptr Source pointer to reallocate.
     * \param size New size of pointer.
     * \param allocator Allocator to use if src_ptr is null.
     *
     * \return Reallocated pointer.
     *
     */
    void* reallocate(void* src_ptr, size_t size, Allocator allocator);

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
    size_t getSize(void* ptr) const;

  private:
    ResourceManager();

    ~ResourceManager() = default;

    ResourceManager (const ResourceManager&) = delete;
    ResourceManager& operator= (const ResourceManager&) = delete;

    strategy::AllocationStrategy* findAllocatorForPointer(void* ptr);
    strategy::AllocationStrategy* findAllocatorForId(int id);
    strategy::AllocationStrategy* getAllocationStrategy(const std::string& name);

    int getNextId() noexcept;

    std::string getAllocatorInformation() const noexcept;

    static ResourceManager* s_resource_manager_instance;

    std::unordered_map<std::string, strategy::AllocationStrategy* > m_allocators_by_name;
    std::unordered_map<int, strategy::AllocationStrategy* > m_allocators_by_id;

    util::AllocationMap m_allocations;

    strategy::AllocationStrategy* m_default_allocator;

    std::unordered_map<resource::MemoryResourceType, strategy::AllocationStrategy*, resource::MemoryResourceTypeHash > m_memory_resources;

    int m_id;

    std::mutex* m_mutex;

    // Methods that need access to m_allocations to print/filter records
    friend void print_allocator_records(Allocator, std::ostream&);
    friend std::vector<const util::AllocationRecord*> get_allocator_records(Allocator);
};

} // end of namespace umpire

#include "umpire/ResourceManager.inl"

#endif // UMPIRE_ResourceManager_HPP

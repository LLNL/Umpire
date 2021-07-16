//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_DynamicPoolMap_HPP
#define UMPIRE_DynamicPoolMap_HPP

#include "umpire/strategy/allocation_strategy.hpp"
#include "umpire/detail/memory_map.hpp"
#include "umpire/detail/replay.hpp"
#include "umpire/detail/memory_sanitizers.hpp"

#include <map>
#include <tuple>

namespace umpire {
namespace strategy {

namespace {
inline static std::size_t round_up(std::size_t num, std::size_t factor)
{
  return num + factor - 1 - (num - 1) % factor;
}
}

/*!
 * \brief Simple dynamic pool for allocations
 *
 * This AllocationStrategy uses Simpool to provide pooling for allocations of
 * any size. The behavior of the pool can be controlled by two parameters: the
 * initial allocation size, and the minimum allocation size.
 *
 * The initial size controls how large the first piece of memory allocated is,
 * and the minimum size controls the lower bound on all future chunk
 * allocations.
 */
template<typename Memory=memory, bool Tracking=true>
class dynamic_pool_map : 
  public allocation_strategy
{
  private:
    using SizeTuple = std::tuple<std::size_t, bool, std::size_t>;
    using AddressTuple = std::tuple<void*, bool, std::size_t>;
    using AddressMap = detail::memory_map<SizeTuple>;
    using SizeMap = std::multimap<std::size_t, AddressTuple>;

  public:
    /*!
     * \brief Callback heuristic to trigger coalesce of free blocks in pool.
     *
     * The registered heuristic callback function will be called immediately
     * after a deallocation() has completed from the pool.
     */
    using CoalesceHeuristic = std::function<bool (const strategy::dynamic_pool_map<Memory, Tracking>&)>;
    static CoalesceHeuristic percent_releasable(int percentage)
    {
      if ( percentage < 0 || percentage > 100 ) {
        UMPIRE_ERROR("Invalid percentage of " << percentage 
            << ", percentage must be an integer between 0 and 100");
      }

      if ( percentage == 0 ) {
        return [=] (const strategy::dynamic_pool_map<Memory, Tracking>& UMPIRE_UNUSED_ARG(pool)) {
            return false;
        };
      }
      else if ( percentage == 100 ) {
        return [=] (const strategy::dynamic_pool_map<Memory, Tracking>& pool) {
            return (pool.get_current_size() == 0 && pool.getReleasableSize() > 0);
        };
      }

      float f = (float)((float)percentage / (float)100.0);

      return [=] (const strategy::dynamic_pool_map<Memory, Tracking>& pool) {
        // Calculate threshold in bytes from the percentage
        const std::size_t threshold = static_cast<std::size_t>(f * pool.get_actual_size());
        return (pool.getReleasableSize() >= threshold);
      };
    }

    /*!
     * \brief Construct a new dynamic_pool_map.
     *
     * \param name Name of this instance of the dynamic_pool_map
     * \param id Unique identifier for this instance
     * \param initial_alloc_bytes Size the pool initially allocates
     * \param min_alloc_bytes The minimum size of all future allocations
     * \param coalesce_heuristic Heuristic callback function
     * \param align_bytes Number of bytes with which to align allocation sizes
     */
    dynamic_pool_map(
        const std::string& name,
        Memory* allocator,
        const std::size_t initial_alloc_size = (512 * 1024 * 1024),
        const std::size_t min_alloc_size = (1 * 1024 * 1024),
        const std::size_t align_bytes = 16,
        CoalesceHeuristic coalesce_heuristic = percent_releasable(100)) noexcept :
      allocation_strategy{name},
      m_allocator{allocator},
      m_initial_alloc_bytes{round_up(initial_alloc_size, align_bytes)},
      m_min_alloc_bytes{round_up(min_alloc_size, align_bytes)},
      m_align_bytes{align_bytes},
      m_coalesce_heuristic{coalesce_heuristic},
      m_used_map{},
      m_free_map{},
      m_actual_bytes{round_up(m_initial_alloc_bytes, m_align_bytes)}
    {
      const std::size_t bytes{round_up(m_initial_alloc_bytes, align_bytes)};
#if defined(UMPIRE_ENABLE_BACKTRACE)
      {
        umpire::util::backtrace bt{};
        umpire::util::backtracer<>::get_backtrace(bt);
        UMPIRE_LOG(Info, "actual_size: " << bytes << " (prev: 0) " << umpire::util::backtracer<>::print(bt));
      }
#endif
      insertFree(m_allocator->allocate(bytes), bytes, true, bytes);
    }

    /*!
     * \brief Destructs the dynamic_pool_map.
     */
    ~dynamic_pool_map()
    {
      // Get as many whole blocks as possible in the m_free_map
      mergeFreeBlocks();

      // Free any unused blocks
      for (auto &rec : m_free_map) {
        const std::size_t bytes{rec.first};
        void *addr;
        bool is_head;
        std::size_t whole_bytes;
        std::tie(addr, is_head, whole_bytes) = rec.second;
        // Deallocate if this is a whole block
        if (is_head && bytes == whole_bytes)
          deallocateBlock(addr, bytes);
      }

      if (m_used_map.size() == 0) {
        UMPIRE_ASSERT(m_actual_bytes == 0);
      }
    }

    dynamic_pool_map(const dynamic_pool_map&) = delete;

    void* allocate(std::size_t n) override
    {
      UMPIRE_LOG(Debug, "(bytes=" << n << ")");

      const std::size_t rounded_bytes = round_up(n, m_align_bytes);
      void* ptr{nullptr};

      // Check if the previous block is a match
      const SizeMap::const_iterator iter{findFreeBlock(rounded_bytes)};

      if (iter != m_free_map.end()) {
        // Found this acceptable address pair
        bool is_head;
        std::size_t whole_bytes;
        std::tie(ptr, is_head, whole_bytes) = iter->second;

        // Add used map
        insertUsed(ptr, rounded_bytes, is_head, whole_bytes);

        // Remove the entry from the free map
        const std::size_t free_size{iter->first};
        m_free_map.erase(iter);

        const std::size_t left_bytes{free_size - rounded_bytes};

        if (left_bytes > 0) {
          insertFree(static_cast<unsigned char*>(ptr) + rounded_bytes, left_bytes,
                    false, whole_bytes);
        }
      } else {
        // Allocate new block -- note this does not check whether alignment is met
        const std::size_t min_block_size =
          ( m_actual_bytes == 0 ) ? m_initial_alloc_bytes : m_min_alloc_bytes;

        const std::size_t alloc_bytes{std::max(rounded_bytes, min_block_size)};
        ptr = allocateBlock(alloc_bytes);

        // Add used
        insertUsed(ptr, rounded_bytes, true, alloc_bytes);

        const std::size_t left_bytes{alloc_bytes - rounded_bytes};

        // Add free
        if (left_bytes > 0)
          insertFree(static_cast<unsigned char*>(ptr) + rounded_bytes, left_bytes,
                    false, alloc_bytes);
      }

      UMPIRE_UNPOISON_MEMORY_REGION(m_allocator, ptr, n);

      if constexpr(Tracking) {
        return this->track_allocation(this, ptr, n);

      } else {
        return ptr;
      }
    }

    void deallocate(void* ptr) override
    {
      UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
      UMPIRE_ASSERT(ptr);

      auto iter = m_used_map.find(ptr);

      if (iter->second) {
        // Fast way to check if key was found

        std::size_t bytes;
        bool is_head;
        std::size_t whole_bytes;
        std::tie(bytes, is_head, whole_bytes) = *iter->second;

        // Insert in free map
        insertFree(ptr, bytes, is_head, whole_bytes);

        // Remove from used map
        m_used_map.erase(iter);

        UMPIRE_POISON_MEMORY_REGION(m_allocator, ptr, bytes);
      } else {
        UMPIRE_ERROR("Cound not found ptr = " << ptr);
      }

      if constexpr(Tracking) {
        this->untrack_allocation(ptr);
      }

      if (m_coalesce_heuristic(*this)) {
        UMPIRE_LOG(Debug, this
                  << " heuristic function returned true, calling coalesce()");
        do_coalesce();
      }
    }

    void release()
    {
      UMPIRE_LOG(Debug, "()");

      // Coalesce first so that we are able to release the most memory possible
      mergeFreeBlocks();

      // Free any blocks with is_head
      releaseFreeBlocks();

      // NOTE This differs from coalesce above in that it does not reallocate a
      // free block to keep actual size the same.
    }

    std::size_t get_actual_size() const noexcept override {
      return m_actual_bytes;
    }

    camp::resources::Platform get_platform() noexcept override
    {
      return m_allocator->get_platform();
    }

    /*!
     * \brief Returns the number of bytes of unallocated data held by this pool
     * that could be immediately released back to the resource.
     *
     * A memory pool has a set of blocks that are not leased out to the
     * application as allocations. Allocations from the resource begin as a
     * single chunk, but these could be split, and only the first chunk can be
     * deallocated back to the resource immediately.
     *
     * \return The total number of bytes that are immediately releasable.
     */
    std::size_t getReleasableSize() const noexcept
    {
      std::size_t releasable_bytes{0};
      for (auto& rec : m_free_map) {
        const std::size_t bytes{rec.first};
        void* ptr;
        bool is_head;
        std::size_t whole_bytes;
        std::tie(ptr, is_head, whole_bytes) = rec.second;
        if (is_head && bytes == whole_bytes) releasable_bytes += bytes;
      }

      return releasable_bytes;
    }

    /*!
     * \brief Return the number of free memory blocks that the pools holds.
     */
    std::size_t getFreeBlocks() const noexcept
    {
      return m_free_map.size();
    }

    /*!
     * \brief Return the number of used memory blocks that the pools holds.
     */
    std::size_t getInUseBlocks() const noexcept
    {
      return m_used_map.size();
    }

    /*!
     * \brief Return the number of memory blocks -- both leased to application
     * and internal free memory -- that the pool holds.
     */
    std::size_t getBlocksInPool() const noexcept
    {
      const std::size_t total_blocks{getFreeBlocks() + getInUseBlocks()};
      return total_blocks;
    }

    /*!
     * \brief Get the largest allocatable number of bytes from pool before
     * the pool will grow.
     *
     * return The largest number of bytes that may be allocated without 
     * causing pool growth
     */
    std::size_t getLargestAvailableBlock() noexcept
    {
      std::size_t largest_block{0};

      mergeFreeBlocks();

      for (auto& rec : m_free_map) {
        const std::size_t bytes{rec.first};
        if (bytes > largest_block) largest_block = bytes;
      }

      UMPIRE_LOG(Debug, "() returning " << largest_block);
      return largest_block;
    }

    /*!
     * \brief Merge as many free records as possible, release all possible free
     * blocks, then reallocate a chunk to keep the actual size the same.
     */
    void coalesce()
    {
      // Coalesce differs from release in that it puts back a single block of the size it released
      UMPIRE_REPLAY("\"event\": \"coalesce\", \"payload\": { \"allocator_name\": \"" << get_name() << "\" }");
      do_coalesce();
    }

  private:
    /*!
     * \brief Allocate from m_allocator.
     */
    void* allocateBlock(std::size_t bytes)
    {
      void* ptr{nullptr};
      try {
    #if defined(UMPIRE_ENABLE_BACKTRACE)
        {
          umpire::util::backtrace bt{};
          umpire::util::backtracer<>::get_backtrace(bt);
          UMPIRE_LOG(Info, "actual_size: " << (m_actual_bytes+bytes) 
            << " (prev: " << m_actual_bytes << ") " 
            << umpire::util::backtracer<>::print(bt));
        }
    #endif
        ptr = m_allocator->allocate(bytes);
      } catch (...) {
        UMPIRE_LOG(Error,
                  "\n\tMemory exhausted at allocation resource. "
                  "Attempting to give blocks back.\n\t"
                  << get_current_size() << " Allocated to pool, "
                  << getFreeBlocks() << " Free Blocks, "
                  << getInUseBlocks() << " Used Blocks\n"
          );
        mergeFreeBlocks();
        releaseFreeBlocks();
        UMPIRE_LOG(Error,
                  "\n\tMemory exhausted at allocation resource.  "
                  "\n\tRetrying allocation operation: "
                  << get_current_size() << " Bytes still allocated to pool, "
                  << getFreeBlocks() << " Free Blocks, "
                  << getInUseBlocks() << " Used Blocks\n"
          );
        try {
          ptr = m_allocator->allocate(bytes);
          UMPIRE_LOG(Error,
                    "\n\tMemory successfully recovered at resource.  Allocation succeeded\n"
            );
        }
        catch (...) {
          UMPIRE_LOG(Error,
                    "\n\tUnable to allocate from resource even after giving back free blocks.\n"
                    "\tThrowing to let application know we have no more memory: "
                    << get_current_size() << " Bytes still allocated to pool\n"
                    << getFreeBlocks() << " Partially Free Blocks, "
                    << getInUseBlocks() << " Used Blocks\n"
            );
          throw;
        }
      }

      UMPIRE_POISON_MEMORY_REGION(m_allocator, ptr, bytes);

      // Add to count
      m_actual_bytes += bytes;

      return ptr;
    }

    /*!
     * \brief Deallocate from m_allocator.
     */
    void deallocateBlock(void* ptr, std::size_t bytes)
    {
      m_actual_bytes -= bytes;
      m_allocator->deallocate(ptr);
    }

    /*!
     * \brief Insert a block to the used map.
     */
    void insertUsed(void* addr, std::size_t bytes, bool is_head,
                    std::size_t whole_bytes)
    {
      m_used_map.insert(std::make_pair(addr, std::make_tuple(bytes, is_head,
                                                            whole_bytes)));
    }

    /*!
     * \brief Insert a block to the free map.
     */
    void insertFree(void* addr, std::size_t bytes, bool is_head,
                    std::size_t whole_bytes)
    {
      m_free_map.insert(std::make_pair(bytes, std::make_tuple(addr, is_head,
                                                              whole_bytes)));
    }

    /*!
     * \brief Find a free block with (length <= bytes) as close to bytes in
     * length as possible.
     */
    SizeMap::const_iterator findFreeBlock(std::size_t bytes) const
    {
      SizeMap::const_iterator iter{m_free_map.upper_bound(bytes)};

      if (iter != m_free_map.begin()) {
        // Back up iterator
        --iter;
        const std::size_t test_bytes{iter->first};
        if (test_bytes < bytes) {
          // Too small, reset iterator to what upper_bound returned
          ++iter;
        }
      }

      return iter;
    }

    /*!
     * \brief Merge all contiguous blocks in m_free_map.
     *
     * NOTE This method is rather expensive, but critical to avoid pool growth
     */
    void mergeFreeBlocks()
    {
      if (m_free_map.size() < 2) return;

      using PointerMap = std::map<void*, SizeTuple>;

      UMPIRE_LOG(Debug, "() Free blocks before: " << getFreeBlocks());

      // Make a free block map from pointers -> size pairs
      PointerMap free_pointer_map;

      for (auto& rec : m_free_map) {
        const std::size_t bytes{rec.first};
        void* ptr;
        bool is_head;
        std::size_t whole_bytes;
        std::tie(ptr, is_head, whole_bytes) = rec.second;
        free_pointer_map.insert(
          std::make_pair(ptr, std::make_tuple(bytes, is_head, whole_bytes)));
      }

      // this map is iterated over from low to high in terms of key = pointer address.
      // Colaesce these...

      auto it = free_pointer_map.begin();
      auto next_it = free_pointer_map.begin(); ++next_it;
      auto end = free_pointer_map.end();

      while (next_it != end) {
        const unsigned char* this_addr{static_cast<unsigned char*>(it->first)};
        std::size_t this_bytes, this_whole_bytes;
        bool this_is_head;
        std::tie(this_bytes, this_is_head, this_whole_bytes) = it->second;

        const unsigned char* next_addr{static_cast<unsigned char*>(next_it->first)};
        std::size_t next_bytes, next_whole_bytes;
        bool next_is_head;
        std::tie(next_bytes, next_is_head, next_whole_bytes) = next_it->second;

        // Check if we can merge *it and *next_it
        const bool contiguous{this_addr + this_bytes == next_addr};
        if (contiguous && !next_is_head) {
          UMPIRE_ASSERT(this_whole_bytes == next_whole_bytes);
          std::get<0>(it->second) += next_bytes;
          next_it = free_pointer_map.erase(next_it);
        } else {
          ++it;
          ++next_it;
        }
      }

      // Now the external map may have shrunk, so rebuild the original map
      m_free_map.clear();
      for (auto& rec : free_pointer_map) {
        void* ptr{rec.first};
        std::size_t bytes, whole_bytes;
        bool is_head;
        std::tie(bytes, is_head, whole_bytes) = rec.second;
        insertFree(ptr, bytes, is_head, whole_bytes);
      }

      UMPIRE_LOG(Debug, "() Free blocks after: " << getFreeBlocks());
    }

    /*!
     * \brief Release blocks from m_free_map that have is_head = true and return
     * the amount of memory released.
     */
    std::size_t releaseFreeBlocks()
    {
      UMPIRE_LOG(Debug, "()");

      std::size_t released_bytes{0};

      auto it = m_free_map.cbegin();
      auto end = m_free_map.cend();

      while (it != end) {
        const std::size_t bytes{it->first};
        void* ptr;
        bool is_head;
        std::size_t whole_bytes;
        std::tie(ptr, is_head, whole_bytes) = it->second;
        if (is_head && bytes == whole_bytes) {
          released_bytes += bytes;
          deallocateBlock(ptr, bytes);
          it = m_free_map.erase(it);
        } else {
          ++it;
        }
      }

    #if defined(UMPIRE_ENABLE_BACKTRACE)
      if (released_bytes > 0) {
        umpire::util::backtrace bt{};
        umpire::util::backtracer<>::get_backtrace(bt);
        UMPIRE_LOG(Info, "actual_size: " << m_actual_bytes 
          << " (prev: " << (m_actual_bytes+released_bytes) 
          << ") " << umpire::util::backtracer<>::print(bt));
      }
    #endif

      return released_bytes;
    }

    void do_coalesce()
    {
      mergeFreeBlocks();
      // Now all possible the free blocks that could be merged have been

      // Only release and create new block if more than one block is present
      if (m_free_map.size() > 1) {
        const std::size_t released_bytes{releaseFreeBlocks()};
        // Deallocated and removed released_bytes from m_free_map

        // If this removed anything from the map, re-allocate a single large chunk and insert to free map
        if (released_bytes > 0) {
          void* const ptr{allocateBlock(released_bytes)};
          insertFree(ptr, released_bytes, true, released_bytes);
        }
      }
    }    

    Memory* m_allocator;
    const std::size_t m_initial_alloc_bytes;
    const std::size_t m_min_alloc_bytes;
    const std::size_t m_align_bytes;
    const CoalesceHeuristic m_coalesce_heuristic;
    AddressMap m_used_map;
    SizeMap m_free_map;
    std::size_t m_actual_bytes;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_dynamic_pool_map_HPP

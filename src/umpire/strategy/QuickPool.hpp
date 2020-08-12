//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_PoolMap_HPP
#define UMPIRE_PoolMap_HPP

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/MemoryMap.hpp"

#include <functional>
#include <map>
#include <tuple>
#include <unordered_map>

namespace umpire {

class Allocator;

namespace util {

class FixedMallocPool;

}

namespace strategy {

class QuickPool :
  public AllocationStrategy
{
  public:
    using Pointer = void*;
    using CoalesceHeuristic = std::function<bool (const strategy::QuickPool& )>;

    static CoalesceHeuristic percent_releasable(int percentage);

    QuickPool(
        const std::string& name,
        int id,
        Allocator allocator,
        const std::size_t initial_alloc_size = (512 * 1024 * 1024),
        const std::size_t min_alloc_size = (1 * 1024 * 1024),
        CoalesceHeuristic coalesce_heuristic = percent_releasable(100)) noexcept;

    ~QuickPool();

    QuickPool(const QuickPool&) = delete;

    void* allocate(std::size_t bytes) override;
    void deallocate(void* ptr) override;
    void release() override;

    std::size_t getActualSize() const noexcept override;
    std::size_t getReleasableSize() const noexcept;

    Platform getPlatform() noexcept override;

    void coalesce() noexcept;

  private:
    struct Chunk;

    template <typename Value>
    class pool_allocator {
      public:
        using value_type = Value;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        pool_allocator() :
          pool{new util::FixedMallocPool{sizeof(Value)}} {}

        ~pool_allocator()
	{
	  if (pool != nullptr) 
	    delete(pool); 
	}

        /// BUG: Only required for MSVC
        template<typename U>
        pool_allocator(const pool_allocator<U>& other):
          pool{other.pool}
        {}

        Value* allocate(std::size_t n) {
          return static_cast<Value*>(pool->allocate(n));
        }

        void deallocate(Value* data, std::size_t)
        {
          pool->deallocate(data);
        }

      util::FixedMallocPool* pool;
    };

    using PointerMap = std::unordered_map<void*, Chunk*>;
    using SizeMap = std::multimap<std::size_t, Chunk*, std::less<std::size_t>, pool_allocator<std::pair<const std::size_t, Chunk*>>>;

    struct Chunk {
      Chunk(void* ptr, std::size_t s, std::size_t cs) :
        data{ptr}, size{s}, chunk_size{cs} {}

      void* data{nullptr};
      std::size_t size{0};
      std::size_t chunk_size{0};
      bool free{true};
      Chunk* prev{nullptr};
      Chunk* next{nullptr};
      SizeMap::iterator size_map_it;
    };

    PointerMap m_pointer_map;
    SizeMap m_size_map;

    util::FixedMallocPool m_chunk_pool;

    strategy::AllocationStrategy* m_allocator;

    CoalesceHeuristic m_should_coalesce;

    const std::size_t m_initial_alloc_bytes;
    const std::size_t m_min_alloc_bytes;

    std::size_t m_actual_bytes{0};
    std::size_t m_releasable_bytes{0};
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_Pool_HPP

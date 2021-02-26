//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef __HOST_SHARED_MEMORY_RESOURCE_HPP__
#define __HOST_SHARED_MEMORY_RESOURCE_HPP__

#include <limits>
#include <string>
#include <thread>

#include <string.h>           // strerror
#include <sys/mman.h>         // mmap
#include <sys/stat.h>         // For mode constants, fstat
#include <fcntl.h>            // For O_* constants
#include <unistd.h>           // ftruncate, fstat
#include <sys/types.h>        // ftruncate, fstat

#include <pthread.h>

#include "umpire/resource/HostSharedMemoryResource.hpp"
#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace resource {

class HostSharedMemoryResource::impl {

  struct SharedMemoryBlock {
    std::size_t next_block_off;  // Offset == 0 is same as nullptr
    std::size_t name_offset;
    std::size_t memory_offset;
    std::size_t size;               // Includes header+name+memory
    std::size_t reference_count;
  };

  struct SharedMemorySegmentHeader {
    uint32_t init_flag;
    uint32_t reserved;
    pthread_mutex_t mutex;
    std::size_t segment_size;    // Full segment size, excluding this header
    std::size_t in_use;
    std::size_t high_watermark;
    std::size_t free_blocks_off;
    std::size_t used_blocks_off;
  };

  public:
    impl(const std::string& name, std::size_t size)
    {
      m_segment_name = ( name[0] != '/' ) ? std::string{"/"} + name : name;
      const uint32_t Initializing{1};
      const uint32_t Initialized{2};

      bool created{ false };
      bool completed{ false };
      int shm_handle{ 0 };
      int err{ 0 };

      UMPIRE_LOG(Debug, " ( " << "name=\"" << name << "\"" << ", size=" << size << ")");

      //
      // SIMPLIFYING ASSUMPTION:
      //
      // This implementation depends upon something else cleaning up what
      // is placed in /dev/shm.  For LC, the epilogue scripts will remove
      // everything that is in /dev/shm so that new job allocations may be
      // assured of having a clean directory.
      //
      while (!completed) { // spin on opening shm
        if ( open_shared_memory_segment(shm_handle, err, (O_RDWR | O_CREAT | O_EXCL) ) ) {
          created = true;
          completed = true;
        }
        else if (err != EEXIST) {
          UMPIRE_ERROR("Failed to create shared memory segment "
                        << m_segment_name << ": " << strerror(err));
        }
        else {
          if (open_shared_memory_segment(shm_handle, err, O_RDWR)) {
            created = false;
            completed = true;
          }
          else if (err != ENOENT) {
            UMPIRE_ERROR("Failed to open shared memory file "
                    << m_segment_name << ": " << strerror(err));
          }
        }
        std::this_thread::yield();
      }

      if (created) {
        if ( 0 != ftruncate(shm_handle, size) ) {
          err = errno;
          UMPIRE_ERROR("Failed to set size for shared memory segment "
                          << m_segment_name << ": " << strerror(err));
        }

        map_shared_memory_segment(shm_handle);

        __atomic_store_n(&shared_mem_header->init_flag, Initializing, __ATOMIC_SEQ_CST);

        pthread_mutexattr_t mattr;
        if ( ( err = pthread_mutexattr_init(&mattr) ) != 0 ) {
          UMPIRE_ERROR("Failed to initialize mutex attributes for shared memory segment "
                          << m_segment_name << ": " << strerror(err));
        }

        if ( ( err = pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED) ) != 0 ) {
          UMPIRE_ERROR("Failed to set shared atributes for mutex for shared memory segment "
                          << m_segment_name << ": " << strerror(err));
        }

        if ( ( err = pthread_mutex_init(&shared_mem_header->mutex, &mattr) ) != 0 ) {
          UMPIRE_ERROR("Failed to initialize mutex for shared memory segment "
                          << m_segment_name << ": " << strerror(err));
        }

        shared_mem_header->segment_size = size;

        pointer_to_offset(&shared_mem_header[1], shared_mem_header->free_blocks_off);
        pointer_to_offset(nullptr, shared_mem_header->used_blocks_off);
        SharedMemoryBlock* block_ptr;
        offset_to_pointer(shared_mem_header->free_blocks_off, block_ptr);
        pointer_to_offset(nullptr, block_ptr->next_block_off);
        pointer_to_offset(nullptr, block_ptr->name_offset);
        block_ptr->size = size - sizeof(SharedMemorySegmentHeader);
        block_ptr->reference_count = 0;

        __atomic_store_n(&shared_mem_header->init_flag, Initialized, __ATOMIC_SEQ_CST);
      }
      else {
        // Wait for the file size to change
        off_t filesize{0};
        while ( filesize == 0 ) {
          struct stat st;

          if ( fstat(shm_handle, &st) < 0 ) {
            err = errno;
            UMPIRE_ERROR("Failed fstat for shared memory segment "
                          << m_segment_name << ": " << strerror(err));
          }
          filesize = st.st_size;
          std::this_thread::yield();
        }

        map_shared_memory_segment(shm_handle);

        uint32_t value{ __atomic_load_n(&shared_mem_header->init_flag, __ATOMIC_SEQ_CST) };

        // Wait for the memory segment header to be initialized
        while ( value != Initialized ) {
          std::this_thread::yield();
          value = __atomic_load_n(&shared_mem_header->init_flag, __ATOMIC_SEQ_CST);
        }
      }
    }

    ~impl()
    {
    }

    void* allocate(const std::string& name, std::size_t requested_size )
    {
      int err{0};
      void* ptr{nullptr};
      const std::size_t header_size{ (sizeof(SharedMemoryBlock) + m_alignment) & ~(m_alignment - 1) };
      const std::size_t mem_size{ (requested_size + m_alignment) & ~(m_alignment - 1) };
      const std::size_t name_size{ (name.length() + 1 + m_alignment) & ~(m_alignment - 1) };
      const std::size_t adjusted_size{ header_size + mem_size + name_size };

      UMPIRE_LOG(Debug, "(name=\"" << name << ", requested_size=" << requested_size << ")");

      if ( ( err = pthread_mutex_lock(&shared_mem_header->mutex) ) != 0 ) {
        UMPIRE_ERROR("Failed to lock mutex for shared memory segment "
                            << m_segment_name << ": " << strerror(err));
      }

      // First let's see if the allaction already exists
      SharedMemoryBlock* best{ find_existing_allocation(name) };
      SharedMemoryBlock* prev{ nullptr };

      if ( best != nullptr ) {
        best->reference_count++;
      }
      else { // New allocation
        findUsableBlock(best, prev, adjusted_size);

        if ( best != nullptr ) {
          // Split the free block
          splitBlock(best, prev, adjusted_size);

          // Push node to the list of used nodes
          best->next_block_off = shared_mem_header->used_blocks_off;
          pointer_to_offset(best, shared_mem_header->used_blocks_off);

          // Set up block header
          std::size_t block_offset;
          pointer_to_offset(best, block_offset);
          best->memory_offset = block_offset + header_size;
          best->name_offset = best->memory_offset + mem_size;
          best->reference_count = 1;

          char* name_ptr;
          offset_to_pointer(best->name_offset, name_ptr);
          name.copy(name_ptr, name.length(), 0);
          name_ptr[name.length()] = '\0';
        }
      }

      offset_to_pointer(best->memory_offset, ptr);

      pthread_mutex_unlock(&shared_mem_header->mutex);

      if ( ptr == nullptr ) {
        UMPIRE_ERROR("shared memory allocation( bytes = " << requested_size << " ) failed");
      }

      return ptr;
    }

    void deallocate( void* ptr )
    {
      UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

      const std::size_t header_size{ (sizeof(SharedMemoryBlock) + m_alignment) & ~(m_alignment - 1) };
      std::size_t data_offset;
      pointer_to_offset(ptr, data_offset);

      std::size_t block_off{ data_offset - header_size };
      SharedMemoryBlock* block_ptr;
      offset_to_pointer(block_off, block_ptr);

      int err{0};
      if ( ( err = pthread_mutex_lock(&shared_mem_header->mutex) ) != 0 ) {
        UMPIRE_ERROR("Failed to lock mutex for shared memory segment "
                            << m_segment_name << ": " << strerror(err));
      }

      block_ptr->reference_count--;

      if ( block_ptr->reference_count == 0 ) {
        SharedMemoryBlock* curr;
        SharedMemoryBlock* prev{nullptr};

        offset_to_pointer(shared_mem_header->used_blocks_off, curr);

        while ( curr != nullptr && curr != block_ptr) {
          prev = curr;
          offset_to_pointer(curr->next_block_off, curr);
        }

        releaseBlock(block_ptr, prev);
      }

      pthread_mutex_unlock(&shared_mem_header->mutex);
    }

    void* find_pointer_from_name(std::string name)
    {
      void* ptr{nullptr};
      int err{0};

      if ( ( err = pthread_mutex_lock(&shared_mem_header->mutex) ) != 0 ) {
        UMPIRE_ERROR("Failed to lock mutex for shared memory segment "
                            << m_segment_name << ": " << strerror(err));
      }

      // First let's see if the allaction already exists
      SharedMemoryBlock* block_ptr{ find_existing_allocation(name) };

      if ( block_ptr != nullptr ) {
        offset_to_pointer(block_ptr->memory_offset, ptr);
      }

      pthread_mutex_unlock(&shared_mem_header->mutex);
      return ptr;
    }

    std::size_t getCurrentSize() const noexcept;
    std::size_t getHighWatermark() const noexcept;

    Platform getPlatform() noexcept;

  private:
    std::string m_segment_name;
    SharedMemorySegmentHeader* shared_mem_header{nullptr};
    std::size_t m_size{0};
    std::size_t m_alignment{16};

    template <class OFF_T, class PTR_T>
    void offset_to_pointer(OFF_T offset, PTR_T& ptr) {
      if ( offset == 0 ) {
        ptr = nullptr;
      }
      else {
        ptr = reinterpret_cast<PTR_T>(reinterpret_cast<char*>(shared_mem_header) + offset);
      }
    }

    void pointer_to_offset(std::nullptr_t, std::size_t& offset)
    {
        offset = 0;
    }

    template <class PTR_T, class OFF_T>
    void pointer_to_offset(PTR_T ptr, OFF_T& offset)
    {
      char* base{ reinterpret_cast<char*>(shared_mem_header) };

      offset = ptr == nullptr ? 0 : static_cast<OFF_T>(reinterpret_cast<char*>(ptr) - base);
    }

    SharedMemoryBlock* find_existing_allocation( std::string name )
    {
      SharedMemoryBlock* block_ptr;
      offset_to_pointer(shared_mem_header->used_blocks_off, block_ptr);

      while ( block_ptr != nullptr ) {
        char* allocation_name;
        offset_to_pointer(block_ptr->name_offset, allocation_name);
        if ( allocation_name != nullptr && 0 == name.compare(allocation_name) )
          break;
        offset_to_pointer(block_ptr->next_block_off, block_ptr);
      }

      return block_ptr;
    }

    void findUsableBlock(SharedMemoryBlock*& best, SharedMemoryBlock*& prev, std::size_t size)
    {
      best = prev = nullptr;

      SharedMemoryBlock* iter;
      offset_to_pointer(shared_mem_header->free_blocks_off, iter);
      SharedMemoryBlock* iterPrev{nullptr};

      while ( iter != nullptr ) {
        if (iter->size >= size && (best == nullptr || iter->size < best->size) ) {
          best = iter;
          prev = iterPrev;
          if (iter->size == size)
            break;  // Exact match, won't find a better one, look no further
        }
        iterPrev = iter;
        offset_to_pointer(iter->next_block_off, iter);
      }
    }

    void releaseBlock(SharedMemoryBlock* curr, SharedMemoryBlock* prev)
    {
      if (prev)
        prev->next_block_off = curr->next_block_off;
      else
        shared_mem_header->used_blocks_off = curr->next_block_off;

      // Find location to put this block in the freeBlocks list
      prev = nullptr;
      SharedMemoryBlock* iter;
      offset_to_pointer(shared_mem_header->free_blocks_off, iter);

      while ( iter != nullptr && iter < curr) {
        prev = iter;
        offset_to_pointer(iter->next_block_off, iter);
      }
      // Keep track of the successor
      std::size_t next_offset = prev ? prev->next_block_off : shared_mem_header->free_blocks_off;
      SharedMemoryBlock* next;
      offset_to_pointer(next_offset, next);
      std::size_t prev_offset;
      pointer_to_offset(prev, prev_offset);
      std::size_t curr_offset;
      pointer_to_offset(curr, curr_offset);

      // Check if prev and curr can be merged
      if (prev && prev_offset + prev->size == curr_offset) {
        prev->size = prev->size + curr->size;
        curr = prev;
        pointer_to_offset(curr, curr_offset);
      } else if (prev) {
        prev->next_block_off = curr_offset;
      } else {
        shared_mem_header->free_blocks_off = curr_offset;
      }

      // Check if curr and next can be merged
      if ( next && ( curr_offset + curr->size == next_offset ) ) {
        curr->size = curr->size + next->size;
        curr->next_block_off = next->next_block_off;
      } else {
        curr->next_block_off = next_offset;
      }
    }

    void splitBlock(SharedMemoryBlock *&curr, SharedMemoryBlock *&prev, const std::size_t size)
    {
      SharedMemoryBlock *next{nullptr};

      if (curr->size == size) {
        // Keep it
        offset_to_pointer(curr->next_block_off, next);
      } else {
        // Split the block
        std::size_t remaining = curr->size - size;
        std::size_t curr_block_off;
        pointer_to_offset(curr, curr_block_off);

        SharedMemoryBlock *newBlock;
        offset_to_pointer(curr_block_off + size, newBlock);

        newBlock->memory_offset = 0;
        newBlock->name_offset = 0;
        newBlock->next_block_off = curr->next_block_off;
        newBlock->reference_count = 0;
        newBlock->size = remaining;

        next = newBlock;
        curr->size = size;
      }

      if ( prev != nullptr ) {
        std::size_t offset;
        pointer_to_offset(prev, offset);
        prev->next_block_off = offset;
      }
      else {
        std::size_t offset;
        pointer_to_offset(next, offset);
        shared_mem_header->free_blocks_off = offset;
      }
    }

    bool open_shared_memory_segment(int& handle, int& err, int oflag)
    {
      constexpr int omode{ S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH };

      handle = shm_open(m_segment_name.c_str(), oflag, omode);
      err = errno;

      bool rval{ handle >= 0 };

      if ( rval && (oflag & O_CREAT) ) {
        ::fchmod(handle, omode);
      }

      return rval;
    }

    void map_shared_memory_segment(int handle)
    {
      struct ::stat buf;
      if ( 0 != fstat(handle, &buf) ) {
        int err = errno;
        UMPIRE_ERROR("Failed to obtain size of shared object "
                            << m_segment_name << ": " << strerror(err));
      }

      auto size = buf.st_size;

      const int prot { PROT_WRITE | PROT_READ };
      const int flags { MAP_SHARED };

      void* base = mmap(  nullptr
                        , static_cast<std::size_t>(size)
                        , prot
                        , flags
                        , handle
                        , 0);

      if (base == MAP_FAILED) {
        int err = errno;
        UMPIRE_ERROR("Failed to map shared object " << m_segment_name << ": " << strerror(err) );
      }

      shared_mem_header = static_cast<SharedMemorySegmentHeader*>(base);
      m_size   = size;
    }
};

} // end of namespace resource
} // end of namespace umpire
#endif // __HOST_SHARED_MEMORY_RESOURCE_HPP__

//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef __HOST_SHARED_MEMORY_RESOURCE_HPP__
#define __HOST_SHARED_MEMORY_RESOURCE_HPP__

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
    std::size_t next_block_offset;  // Offset == 0 is same as nullptr
    std::size_t prev_block_offset;
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
    SharedMemoryBlock free_memory;
    SharedMemoryBlock allocated_memory;
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

        pointer_to_offset(nullptr, shared_mem_header->free_memory.next_block_offset);
        pointer_to_offset(nullptr, shared_mem_header->free_memory.prev_block_offset);
        pointer_to_offset(nullptr, shared_mem_header->free_memory.name_offset);
        pointer_to_offset(&shared_mem_header[1], shared_mem_header->free_memory.memory_offset);

        shared_mem_header->free_memory.size = size - sizeof(SharedMemorySegmentHeader);
        shared_mem_header->free_memory.reference_count = 0;

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

  SharedMemoryBlock* find_allocation( std::string name )
  {
    SharedMemoryBlock* mblock_ptr{ nullptr };
    return mblock_ptr;
  }

  void* allocate(const std::string& name, std::size_t requested_size)
  {
    std::size_t header_size{ (sizeof(SharedMemoryBlock) + m_alignment) & ~(m_alignment - 1) };
    std::size_t name_size{ (name.length() + 1 + m_alignment) & ~(m_alignment - 1) };
    std::size_t mem_size{ (requested_size + m_alignment) & ~(m_alignment - 1) };
    std::size_t adjusted_size{ header_size + name_size + mem_size };

    int err{0};
    if ( ( err = pthread_mutex_lock(&shared_mem_header->mutex) ) != 0 ) {
      UMPIRE_ERROR("Failed to lock mutex for shared memory segment "
                          << m_segment_name << ": " << strerror(err));
    }

    // First let's see if the allaction already exists
    SharedMemoryBlock* mblock_ptr{ find_allocation(name) };

#if 0
    //
    // Find amount available memory
    //
    shared_mem_header->free_memory.next_block_offset = 0;
    shared_mem_header->free_memory.prev_block_offset = 0;
    shared_mem_header->free_memory.name_offset = 0;
    shared_mem_header->free_memory.size = size - sizeof(SharedMemorySegmentHeader);
    shared_mem_header->free_memory.reference_count = 0;
#endif
    pthread_mutex_unlock(&shared_mem_header->mutex);

    return static_cast<void*>(nullptr);
  }

  void deallocate(void* /* ptr */)
  {
  }

  void* find_pointer_from_name(std::string /* name */)
  {
    return nullptr;
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

    offset = static_cast<OFF_T>(reinterpret_cast<char*>(ptr) - base);
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

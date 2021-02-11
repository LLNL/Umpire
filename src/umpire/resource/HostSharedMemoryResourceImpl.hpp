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

  struct shared_memory_header {
    uint32_t init_flag;
    uint32_t reserved;
    pthread_mutex_t mutex;
    std::size_t segment_size;    // Full segment size, excluding this header
    std::size_t in_use;
    std::size_t high_watermark;
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
        pthread_mutexattr_init(&mattr);
        pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
        pthread_mutex_init(&shared_mem_header->mutex, &mattr);

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

  void* allocate(const std::string& /* name */, std::size_t /* bytes */)
  {
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
  shared_memory_header* shared_mem_header{nullptr};
  std::size_t m_size{0};

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

    shared_mem_header = static_cast<shared_memory_header*>(base);
    m_size   = size;
  }
};

} // end of namespace resource
} // end of namespace umpire
#endif // __HOST_SHARED_MEMORY_RESOURCE_HPP__

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

#include <string.h>           /* strerror */
#include <sys/mman.h>         /* mmap */
#include <sys/stat.h>         /* For mode constants */
#include <fcntl.h>            /* For O_* constants */
#include <unistd.h>           /* ftruncate,fstat */
#include <sys/types.h>        /* ftruncate, fstat*/

#include "umpire/resource/HostSharedMemoryResource.hpp"
#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace resource {

class HostSharedMemoryResource::impl {
  public:
    impl(const std::string& name, std::size_t size) :
      m_segment_name{ std::string{"/dev/shm/"} + name }
    {
      // TODO: Check if the requested size is enough for our metadata

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
      }

      if (completed) {
        UMPIRE_ERROR("Not implemented yet");
      }
    }

#if 0
{
  DeviceAbstraction dev;
  if(created){
      try{
        //If this throws, we are lost
        truncate_device<FileBased>(dev, size, file_like_t());

        //If the following throws, we will truncate the file to 1
        mapped_region        region(dev, read_write, 0, 0, addr);
        boost::uint32_t *patomic_word = 0;  //avoid gcc warning
        patomic_word = static_cast<boost::uint32_t*>(region.get_address());
        boost::uint32_t previous = atomic_cas32(patomic_word, InitializingSegment, UninitializedSegment);

        if(previous == UninitializedSegment){
            try{
              construct_func( static_cast<char*>(region.get_address()) + ManagedOpenOrCreateUserOffset
                            , size - ManagedOpenOrCreateUserOffset, true);
              //All ok, just move resources to the external mapped region
              m_mapped_region.swap(region);
            }
            catch(...){
              atomic_write32(patomic_word, CorruptedSegment);
              throw;
            }
            atomic_write32(patomic_word, InitializedSegment);
        }
        else if(previous == InitializingSegment || previous == InitializedSegment){
            throw interprocess_exception(error_info(already_exists_error));
        }
        else{
            throw interprocess_exception(error_info(corrupted_error));
        }
      }
      catch(...){
        try{
            truncate_device<FileBased>(dev, 1u, file_like_t());
        }
        catch(...){
        }
        throw;
      }
  }
  else{
      if(FileBased){
        offset_t filesize = 0;
        spin_wait swait;
        while(filesize == 0){
            if(!get_file_size(file_handle_from_mapping_handle(dev.get_mapping_handle()), filesize)){
              error_info err = system_error_code();
              throw interprocess_exception(err);
            }
            swait.yield();
        }
        if(filesize == 1){
            throw interprocess_exception(error_info(corrupted_error));
        }
      }

      mapped_region  region(dev, ronly ? read_only : (cow ? copy_on_write : read_write), 0, 0, addr);

      boost::uint32_t *patomic_word = static_cast<boost::uint32_t*>(region.get_address());
      boost::uint32_t value = atomic_read32(patomic_word);

      spin_wait swait;
      while(value == InitializingSegment || value == UninitializedSegment){
        swait.yield();
        value = atomic_read32(patomic_word);
      }

      if(value != InitializedSegment)
        throw interprocess_exception(error_info(corrupted_error));

      construct_func( static_cast<char*>(region.get_address()) + ManagedOpenOrCreateUserOffset
                    , region.get_size() - ManagedOpenOrCreateUserOffset
                    , false);
      //All ok, just move resources to the external mapped region
      m_mapped_region.swap(region);
  }
  if(StoreDevice){
      this->DevHolder::get_device() = boost::move(dev);
  }
}
#endif

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
  std::size_t m_current_size{0};
  std::size_t m_high_watermark{0};
  char* m_base{nullptr};
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

    m_base = static_cast<char*>(base);
    m_size   = size;
  }
};

} // end of namespace resource
} // end of namespace umpire
#endif // __HOST_SHARED_MEMORY_RESOURCE_HPP__
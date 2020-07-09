//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_FileMemoryResource_INL
#define UMPIRE_FileMemoryResource_INL

#include "umpire/resource/FileMemoryResource.hpp"

#include "umpire/util/Macros.hpp"

#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <ctime>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>

namespace umpire {
namespace resource {

int FILE_COUNTER = 0;

FileMemoryResource::FileMemoryResource(
    Platform platform, 
    const std::string& name,
    int id,
    MemoryResourceTraits traits) :
  MemoryResource(name, id, traits),
  m_platform{platform},
  m_size_map{}
{
} 

void* FileMemoryResource::allocate(std::size_t bytes)
{
  if (bytes <= 0) { UMPIRE_ERROR( "Bytes Requested Error: Bytes size is 0"); }

  // Setting File Name And Opening the files
  std::stringstream SS;
  SS << "./umpire_mem_" << getpid() << FILE_COUNTER;
  FILE_COUNTER++;

  int fd = open(SS.str().c_str(), O_RDWR | O_CREAT | O_LARGEFILE, S_IRWXU);
  if (fd == -1) { UMPIRE_ERROR("Opening File Failed: " << strerror(errno)); }

  // Setting Size Of Map File
  std::size_t SIZE = bytes / sysconf(_SC_PAGE_SIZE);
  if(SIZE == 0)
    SIZE = sysconf(_SC_PAGE_SIZE);
  else
    SIZE = (sysconf(_SC_PAGE_SIZE) * SIZE) + (bytes % sysconf(_SC_PAGE_SIZE));
  if (SIZE < bytes) { remove(SS.str().c_str()); UMPIRE_ERROR("Size Setting Failed: Size is not properly allocated"); }
 
  // Truncate file
  int trun{ftruncate64(fd, SIZE)};
  if (trun == -1) { remove(SS.str().c_str()); UMPIRE_ERROR("Truncate Failed: " << strerror(errno)); }

  // Using mmap
  void* ptr{mmap(NULL, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)};
  if (ptr == MAP_FAILED) { remove(SS.str().c_str()); UMPIRE_ERROR("Mmap Failed: " << strerror(errno)); }

  // Storing Information On File
  std::pair <const std::string, std::size_t> INFO = std::make_pair(SS.str(), SIZE);
  m_size_map.insert(ptr, INFO);
  
  close(fd);
  return ptr;
}

void FileMemoryResource::deallocate(void* ptr)
{
  auto iter = m_size_map.find(ptr);
  auto size = iter->second->second;
  const std::string file_name = iter->second->first;
  m_size_map.erase(ptr);

  munmap(ptr, size);

  remove(file_name.c_str());
}

std::size_t FileMemoryResource::getCurrentSize() const noexcept
{
  return 0;
}

std::size_t FileMemoryResource::getHighWatermark() const noexcept
{
  return 0;
}

Platform FileMemoryResource::getPlatform() noexcept
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
#endif // UMPIRE_FileMemoryResource_INL

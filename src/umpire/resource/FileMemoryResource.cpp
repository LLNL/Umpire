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

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <memory>
#include <sstream>
#include <time.h>
#include <iostream>
#include <cstdio>
#include <string>
#include <stdio.h>
#include <utility>

namespace umpire {
namespace resource {

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
  const char * file = std::tmpnam(nullptr);
  std::size_t SIZE = bytes / sysconf(_SC_PAGE_SIZE);
  int fd = open(file, O_RDWR | O_CREAT, S_IRWXU);

  if(SIZE == 0)
    SIZE = sysconf(_SC_PAGE_SIZE);
  else
    SIZE = (sysconf(_SC_PAGE_SIZE) * SIZE) + (bytes % sysconf(_SC_PAGE_SIZE));

  truncate(file, SIZE);
  void* ptr{mmap(NULL, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)};
  
  std::pair <const char *, std::size_t> INFO;
  INFO = std::make_pair(file,SIZE);
  m_size_map.insert(ptr, INFO);

  close(fd);
  return ptr;
}

void FileMemoryResource::deallocate(void* ptr)
{
  auto iter = m_size_map.find(ptr);
  auto size = iter->second->second;
  auto file_name = iter->second->first;
  m_size_map.erase(ptr);

  munmap(ptr, size);

  remove(file_name);
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
//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_FileMemoryResource_HPP
#define UMPIRE_FileMemoryResource_HPP

#include <utility>

#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/MemoryMap.hpp"
#include "umpire/util/Platform.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief File Memory allocator
 *
 * This FileMemoryResource uses mmap to create a file mapping in order to use as
 * additional memory. To create this mapping the function needs to take in the
 * size of memory wanted for the allocation. The set location for the allocation
 * by defult is ./ but can be assigned using enviroment variable
 * "UMPIRE_MEMORY_FILE_DIR"
 *
 * The return should be a pointer location. The same pointer location can be
 * used for the deallocation. Deallocation uses munmap and removes the file
 * associated with the pointer location.
 */
class FileMemoryResource : public MemoryResource {
 public:
  /*!
   * \brief Construct a new FileMemoryResource
   *
   * \param platform Platform of this instance of the FileMemoryResource.
   * \param name Name of this instance of the FileMemoryResource.
   * \param id Id of this instance of the FileMemoryResource.
   * \param traits Traits of this instance of the FileMemoryResource.
   */
  FileMemoryResource(Platform platform, const std::string& name, int id,
                     MemoryResourceTraits traits);

  /*!
   * \brief Dallocates and removes all files created by the code meant for
   * allocations
   */
  ~FileMemoryResource();

  /*!
   * \brief Creates the allocation of size bytes using mmap
   *
   * Does the allocation as follows:
   * 1) Find output file directory for mmap files using UMPIRE_MEMORY_FILE_DIR
   * 2) Create name and create the file index using open
   * 3) Setting Size Of Map File. Size is scaled to a page length on the system.
   * 4) Truncate file using ftruncate64
   * 5) Map file index with mmap
   * 6) Store information about the allocated file into m_size_map
   *
   * \param bytes The requested amount of bytes the user wants to use.
   * Can not be zero or greater than avalable amount of bytes available.
   * \param UMPIRE_MEMORY_FILE_DIR Used to specify where memory is going
   * to be allocated from
   *
   * \return void* Since you are only reciving a pointer location of any
   * size non spcaific to a type you will have to cast it to the desired
   * type if needed.
   */
  void* allocate(std::size_t bytes);

  /*!
   * \brief Deallocates file connected to the pointer
   *
   * Using m_size_map, the pointer is looked up and the file name and size can
   * be returned. With this munmap can be called to deallocated the correct
   * file.
   *
   * \param ptr Pointer location used to look up its information in m_size_map
   */
  void deallocate(void* ptr);

  std::size_t getCurrentSize() const noexcept;
  std::size_t getHighWatermark() const noexcept;

  Platform getPlatform() noexcept;
  static int s_file_counter;

 protected:
  Platform m_platform;

 private:
  /*!
   * \brief Creates a map of the pointers used in the allocation
   *
   * \param std::pair Paring of the file name and the size of the file
   */
  util::MemoryMap<std::pair<const std::string, std::size_t>> m_size_map;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_FileMemoryResource_HPP

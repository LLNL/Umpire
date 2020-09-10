//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>

#include "umpire/resource/BoostMemoryResource.hpp"
#include "umpire/resource/MemoryResource.hpp"
#include "umpire/tpl/boost_1_74_0/boost/interprocess/managed_shared_memory.hpp"
#include "umpire/tpl/boost_1_74_0/boost/interprocess/managed_mapped_file.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace resource {

class BoostMemoryResource::impl {
  public:
    impl(const std::string& name, std::size_t size) :
      m_segment_name{ name }
    {
      m_segment = new boost::interprocess::managed_shared_memory{
                                boost::interprocess::open_or_create
                              , name.c_str()
                              , size };
    }

    ~impl()
    {
      delete m_segment;
      boost::interprocess::shared_memory_object::remove(m_segment_name.c_str());
    }

    void* allocate(const std::string& name, std::size_t bytes)
    {
      char* ptr{ m_segment->find_or_construct<char>(name.c_str())[bytes](char(0)) };

      return static_cast<void*>(ptr);
    }

    void deallocate(void* ptr)
    {
      m_segment->destroy_ptr(ptr);
    }

    std::size_t getCurrentSize() const noexcept;
    std::size_t getHighWatermark() const noexcept;

    Platform getPlatform() noexcept;

  private:
    boost::interprocess::managed_shared_memory* m_segment;
    std::string m_segment_name;
    std::size_t m_current_size{0};
    std::size_t m_high_watermark{0};
};

} // end of namespace resource
} // end of namespace umpire

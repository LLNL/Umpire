//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_CudaUnifiedMemoryResourceFactory_HPP
#define UMPIRE_CudaUnifiedMemoryResourceFactory_HPP

#include "umpire/resource/MemoryResourceFactory.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief Factory class to construct a MemoryResource that uses NVIDIA
 * "unified" memory, accesible from both the CPU and NVIDIA GPUs.
 */
class CudaUnifiedMemoryResourceFactory :
  public MemoryResourceFactory
{
  bool isValidMemoryResourceFor(const std::string& name) noexcept;
  std::shared_ptr<MemoryResource> create(const std::string& name, int id);
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_CudaUnifiedMemoryResourceFactory_HPP

//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_CudaConstantMemoryResourceFactory_HPP
#define UMPIRE_CudaConstantMemoryResourceFactory_HPP

#include "umpire/resource/MemoryResourceFactory.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief Factory class for constructing MemoryResource objects that use GPU
 * memory.
 */
class CudaConstantMemoryResourceFactory :
  public MemoryResourceFactory
{
public:
  bool isValidMemoryResourceFor(const std::string& name) noexcept final override;

  std::unique_ptr<resource::MemoryResource> 
  create(const std::string& name, int id) final override;

  std::string handle() const noexcept final override;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_CudaConstantMemoryResourceFactory_HPP

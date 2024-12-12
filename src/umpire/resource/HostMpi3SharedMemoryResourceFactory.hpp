//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HostMpi3SharedMemoryResourceFactory_HPP
#define UMPIRE_HostMpi3SharedMemoryResourceFactory_HPP

#include "umpire/resource/MemoryResourceFactory.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief Factory class to construct a MemoryResource.
 */
class HostMpi3SharedMemoryResourceFactory : public MemoryResourceFactory {
  bool isValidMemoryResourceFor(const std::string& name) noexcept final override;

  std::unique_ptr<resource::MemoryResource> create(const std::string& name, int id) final override;

  std::unique_ptr<resource::MemoryResource> create(const std::string& name, int id,
                                                   MemoryResourceTraits traits) final override;

  MemoryResourceTraits getDefaultTraits() final override;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_HostMpi3SharedMemoryResourceFactory_HPP

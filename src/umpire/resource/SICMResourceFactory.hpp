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
#ifndef UMPIRE_SICMResourceFactory_HPP
#define UMPIRE_SICMResourceFactory_HPP

#include <vector>

#include "umpire/resource/MemoryResourceFactory.hpp"

namespace umpire {
namespace resource {


/*!
 * \brief Factory class to construct a MemoryResource that uses SICM.
 */
class SICMResourceFactory :
  public MemoryResourceFactory
{
public:
  SICMResourceFactory(const std::string& name, const std::vector<unsigned int>& devices);

private:
  bool isValidMemoryResourceFor(const std::string& name) noexcept;
  resource::MemoryResource* create(const std::string& name, int id);

  const std::string replacement;
  const std::vector<unsigned int> devices;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_SICMResourceFactory_HPP

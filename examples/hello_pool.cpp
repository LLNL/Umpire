//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"

int main(int, char**)
{
  void* data{nullptr};
  auto& rm = umpire::ResourceManager::getInstance();
  auto pool = rm.makeAllocator<umpire::strategy::DynamicPoolList, false>
                              ("pool", rm.getAllocator("HOST"));

  data = static_cast<double*>(pool.allocate(1024*sizeof(double)));
  pool.deallocate(data);

  return 0;
}
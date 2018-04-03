//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"

#include "umpire/TypedAllocator.hpp"

int main(int, char**) {
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator("HOST");

  umpire::TypedAllocator<double> vector_allocator(alloc);

  std::vector< double, umpire::TypedAllocator<double> > my_vector(vector_allocator);

  my_vector.resize(100);

  my_vector[50] = 3.14;

  return 0;
}

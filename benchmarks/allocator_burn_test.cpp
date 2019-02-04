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
#include <chrono>
#include <functional>
#include <iostream>
#include <random>

#include "umpire/config.hpp"

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

void burn(std::string name) {
  std::mt19937 gen(12345678);
  std::uniform_int_distribution<size_t> dist(64, 128);
  auto random_size = std::bind(dist, gen);

  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator(name);

  while (true) {
    void* data = alloc.allocate(random_size());
    alloc.deallocate(data);
  }
}

int main(int, char**) {
  burn("HOST");
}

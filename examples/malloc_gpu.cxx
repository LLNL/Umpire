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
#include "umpire/Umpire.hpp"
#include "umpire/ResourceManager.hpp"

#include <iostream>

int main() {
  const int size = 100;

  umpire::ResourceManager rm = umpire::ResourceManager::getInstance();
  auto space = rm.getSpace("DEVICE");
  double* my_array = static_cast<double*>(rm.allocate(size * sizeof(double), space));

  umpire::free(my_array);
}

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
#include <stdio.h>

#include "umpire/interface/umpire.h"

#define SIZE 1024

int main(int argc, char** argv) {
  umpire_resourcemanager rm;
  umpire_resourcemanager_get_instance(&rm);

  umpire_allocator allocator;
  umpire_resourcemanager_get_allocator_by_name(&rm, "HOST", &allocator);

  double* data = (double*) umpire_allocator_allocate(&allocator, SIZE*sizeof(double));

  printf("Allocated %d bytes using the %s allocator...", (SIZE*sizeof(double)), umpire_allocator_get_name(&allocator));

  umpire_allocator_deallocate(&allocator, data);

  printf("deallocated.\n");

  return 0;
}

//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <stdio.h>

#include "umpire/interface/umpire.h"

#define SIZE 1024

int main() {
  umpire_resourcemanager rm;
  umpire_resourcemanager_get_instance(&rm);

  umpire_allocator allocator;
  umpire_resourcemanager_get_allocator_by_name(&rm, "HOST", &allocator);

  double* data = (double*) umpire_allocator_allocate(&allocator, SIZE*sizeof(double));

  printf("Allocated %lu bytes using the %s allocator...", (SIZE*sizeof(double)), umpire_allocator_get_name(&allocator));

  umpire_allocator_deallocate(&allocator, data);

  printf("deallocated.\n");

  return 0;
}

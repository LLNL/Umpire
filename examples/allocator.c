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
#include <stdio.h>

#include "umpire/umpire.h"

int main(int argc, char* argv[]) {

  (void)(argc);
  (void)(argv);

  umpire_resourcemanager* rm = umpire_resourcemanager_get_instance();
  umpire_allocator* allocator = umpire_resourcemanager_get_allocator(rm, "HOST");

  double* alloc_one;
  double* alloc_two;

  int ELEMS = 100;
  int i;

  printf("Allocating memory on HOST...");
  alloc_one = (double*) umpire_allocator_allocate(allocator, sizeof(double)*ELEMS);
  alloc_two = (double*) umpire_allocator_allocate(allocator, sizeof(double)*ELEMS);

  for (i = 0; i < ELEMS; i++) {
    alloc_one[i] = 1.0*i;
  }

  umpire_resourcemanager_copy(rm, alloc_one, alloc_two);

  for (i = 0; i < ELEMS; i++) {
    if (alloc_one[i] == alloc_two[i]) {
    } else {
      printf("Boooooo!\n");
    }
  }

  printf("Deallocating memory on HOST...");
  umpire_allocator_deallocate(allocator, alloc_one);
  umpire_allocator_deallocate(allocator, alloc_two);
  printf("done.\n");

  return 0;
}

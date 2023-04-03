//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <stdio.h>

#include "umpire/interface/c_fortran/umpire.h"

#define SIZE 1024

int main() {
  /* _sphinx_tag_tut_c_get_allocator_start */
  umpire_resourcemanager rm;
  umpire_resourcemanager_get_instance(&rm);

  umpire_allocator allocator;
  umpire_resourcemanager_get_allocator_by_name(&rm, "HOST", &allocator);
  /* _sphinx_tag_tut_c_get_allocator_end */

  /* _sphinx_tag_tut_c_allocate_start */
  double* data = (double*) umpire_allocator_allocate(&allocator, SIZE*sizeof(double));

  printf("Allocated %lu bytes using the %s allocator...", (SIZE*sizeof(double)), umpire_allocator_get_name(&allocator));

  umpire_allocator_deallocate(&allocator, data);
  /* _sphinx_tag_tut_c_allocate_end */

  printf("deallocated.\n");

  return 0;
}

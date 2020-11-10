//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <stdio.h>

#include "umpire/interface/umpire.h"

#define SIZE 1024

void allocate_and_deallocate(const char* resource)
{
  /* _sphinx_tag_tut_create_allocator_start */
  umpire_resourcemanager rm;
  umpire_resourcemanager_get_instance(&rm);

  umpire_allocator allocator;
  umpire_resourcemanager_get_allocator_by_name(&rm, resource, &allocator);
  /* _sphinx_tag_tut_create_allocator_end */

  /* _sphinx_tag_tut_allocate_start */
  double* data = (double*) umpire_allocator_allocate(&allocator, SIZE*sizeof(double));
  /* _sphinx_tag_tut_allocate_end */

  printf("Allocated %lu bytes using the %s allocator...",
      (SIZE*sizeof(double)), umpire_allocator_get_name(&allocator));

  umpire_allocator_deallocate(&allocator, data);

  printf("deallocated.\n");
}

int main()
{
  /* _sphinx_tag_tut_resource_types_start */
  allocate_and_deallocate("HOST");

#if defined(UMPIRE_ENABLE_DEVICE)
  allocate_and_deallocate("DEVICE");
#endif
#if defined(UMPIRE_ENABLE_UM)
  allocate_and_deallocate("UM");
#endif
#if defined(UMPIRE_ENABLE_PINNED)
  allocate_and_deallocate("PINNED");
#endif
  /* _sphinx_tag_tut_resource_types_end */

  return 0;
}

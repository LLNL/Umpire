#include <stdio.h>

#include "umpire/umpire.h"

int main(int argc, char* argv[]) {

  (void)(argc);
  (void)(argv);

  UMPIRE_resourcemanager* rm = UMPIRE_resourcemanager_get();
  UMPIRE_allocator* allocator = UMPIRE_resourcemanager_get_allocator(rm, "HOST");

  double* alloc_one;
  double* alloc_two;

  int ELEMS = 100;
  int i;

  printf("Allocating memory on HOST...");
  alloc_one = (double*) UMPIRE_allocator_allocate(allocator, sizeof(double)*ELEMS);
  alloc_two = (double*) UMPIRE_allocator_allocate(allocator, sizeof(double)*ELEMS);

  for (i = 0; i < ELEMS; i++) {
    alloc_one[i] = 1.0*i;
  }

  UMPIRE_resourcemanager_copy(rm, alloc_one, alloc_two);

  for (i = 0; i < ELEMS; i++) {
    if (alloc_one[i] == alloc_two[i]) {
    } else {
      printf("Boooooo!\n");
    }
  }

  printf("Deallocating memory on HOST...");
  UMPIRE_allocator_deallocate(allocator, alloc_one);
  UMPIRE_allocator_deallocate(allocator, alloc_two);
  printf("done.\n");

  return 0;
}

#include "umpire/umpire.h"

int main(int argc, char* argv[]) {

  UMPIRE_resourcemanager* rm = UMPIRE_resourcemanager_get();
  UMPIRE_allocator* allocator = UMPIRE_resourcemanager_get_allocator(rm, "HOST");

  double* allocation;

  printf("allocation = %x\n", allocation);

  printf("Allocating memory on HOST...");
  allocation = (double*) UMPIRE_allocator_allocate(allocator, sizeof(double)*100);
  printf("done @%x.\n", allocation);

  printf("Deallocating memory on HOST...");
  UMPIRE_allocator_deallocate(allocator, allocation);
  printf("done.\n");

  printf("allocation = %x\n", allocation);

  allocation[0] = 1.0;



  return 0;
}

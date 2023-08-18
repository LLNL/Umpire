#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"

void* test_realloc(void* ptr, size_t size, size_t old_size, umpire::Allocator alloc)
{
  void* ret = (void*) alloc.allocate(size);

  hipMemcpy(ret, ptr, old_size, hipMemcpyDeviceToDevice);

  //Must sync or else fails
  //To break reproducer, comment out this sync call
  //hipDeviceSynchronize();

  alloc.deallocate(ptr);
  return ret;
}

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  // Using a quickpool constructed from Unified Memory works.
  umpire::Allocator aloc = rm.getAllocator("DEVICE");
  umpire::Allocator *allocator = new umpire::Allocator(
                    rm.makeAllocator<umpire::strategy::QuickPool>(
                    "permanent_pool", aloc));

  int* ptr = nullptr;
  int MAX_SIZE = 2048;
  for (int start_size = 1; start_size <= MAX_SIZE; start_size *= 2)
  {
      printf("===== Check realloc from %i to %i ====\n", start_size / 2 , start_size);
      if (start_size == 1)
      {
          ptr = (int*) allocator->allocate(sizeof(int) * start_size);
      }
      else
      {
          //Instead of using Umpire, call test_realloc which doesn't call Umpire
          //ptr = (int*) rm.reallocate(ptr, sizeof(int) * start_size);
          ptr = (int*) test_realloc(ptr, sizeof(int) * start_size,
                                   (sizeof(int) * start_size) / 2, *allocator);
      }
      for (int index = 0; index < start_size / 2; index++)
      {
          // Check old elements after alloc/reallocate -- this fails with QuickPool + DEVICE.
          if (ptr[index] != index) { printf("ERROR: ptr[%i] invalid\n", index); }
      }
      printf("- Setting elements for next round.\n");
      for (int index = 0; index < start_size; index++)
      {
          // Set new elements before next allocate.
          ptr[index] = index;
      }
      for (int index = 0; index < start_size; index++)
      {
          // Check new elements after setting - this seems to work fine.
          if (ptr[index] != index) { printf("ERROR: ptr[%i] not set\n", index); }
      }
  }

  return 0;
}

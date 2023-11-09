#include <stddef.h>
#include <cstdint>

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"

size_t random_name_size(const char *str, size_t size)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");
  void* data{nullptr};

  data = alloc.allocate(str, size);
  alloc.deallocate(data);

  return 0;  
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  random_name_size(reinterpret_cast<const char *>(Data), Size);
  return 0;
}

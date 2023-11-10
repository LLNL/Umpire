#include <assert.h>

#include <cifuzz/cifuzz.h>
#include <fuzzer/FuzzedDataProvider.h>
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"

size_t random_size_name(int size, std::string str)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc = rm.getAllocator("HOST");
  rm.addAlias(str, alloc);

  auto alloc_alias = rm.getAllocator(str);
  assert(alloc_alias.getId() == alloc.getId());
  assert(alloc_alias.getName() == alloc.getName());
  assert(alloc_alias.getName() != str);

  void* data = alloc_alias.allocate(size);
  alloc_alias.deallocate(data);

  return 0;
}

FUZZ_TEST_SETUP() {
  // Perform any one-time setup required by the FUZZ_TEST function.
}

FUZZ_TEST(const uint8_t *data, size_t size) {
  FuzzedDataProvider fuzzed_data(data, size);
  int my_int = fuzzed_data.ConsumeIntegral<int8_t>();
  std::string my_string = fuzzed_data.ConsumeRandomLengthString();

  int res{0};
  if (my_string.length() > 0)
    res = random_size_name(std::abs(my_int % 1024), my_string);
  assert(res == 0);
}

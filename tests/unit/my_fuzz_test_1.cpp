#include <assert.h>

#include <cifuzz/cifuzz.h>
#include <fuzzer/FuzzedDataProvider.h>
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"

size_t random_size_name(int size, std::string str)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");
  void* data{nullptr};

  data = alloc.allocate(str, size);
  alloc.deallocate(data);

  return 0;
}

FUZZ_TEST_SETUP() {
  // Perform any one-time setup required by the FUZZ_TEST function.
}

FUZZ_TEST(const uint8_t *data, size_t size) {
  FuzzedDataProvider fuzzed_data(data, size);
  int my_int = fuzzed_data.ConsumeIntegral<int8_t>();
  std::string my_string = fuzzed_data.ConsumeRandomLengthString();

  random_size_name(my_int, my_string);

  // assert(res != -1);
  // If you want to know more about writing fuzz tests you can check out the
  // example projects at https://github.com/CodeIntelligenceTesting/cifuzz/tree/main/examples
  // or have a look at our docs at https://docs.code-intelligence.com/
}

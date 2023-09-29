#include <stddef.h>
#include <cstdint>

size_t find_newline(const char *str)
{
  size_t loc = 0;
  while (str[loc] != '\n') {
    ++loc;
  }
  return loc;  
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  find_newline(reinterpret_cast<const char *>(Data));
  Size++;
  return 0;
}

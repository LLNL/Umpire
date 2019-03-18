#include "FixedSizePool.hpp"
#include "DynamicSizePool.hpp"
#include "StdAllocator.hpp"

typedef FixedSizePool<int, StdAllocator> FPA;
typedef DynamicSizePool<StdAllocator, StdAllocator> DPA;

int main() {
  FPA& fpa = FPA::getInstance();
  DPA& dpa = DPA::getInstance();
  return 0;
}

#include "DynamicSizePool.hpp"
#include "StdAllocator.hpp"
#include "AllocatorTest.hpp"

#include <cassert>
#include <cstdlib>
#include <stack>

#if defined(REDEFINE_NEW)
#error New is used internally in std::stack, so this option breaks the tests below
#endif

// Test with the generic AllocatorType that can be changed at compile
// time (see AllocatorTest.hpp)
typedef DynamicSizePool<AllocatorType> PoolType;

std::size_t rand_int(const std::size_t min, const std::size_t max) {
  return min + static_cast<float>(std::rand()) / RAND_MAX * (max - min);
}

int main() {
  std::srand(0);

  PoolType &pa = PoolType::getInstance();
  std::stack<std::size_t> sizeStack;
  std::stack<void*> ptrStack;

  const int numIter = (1 << 2);
  for (int i = 0; i < numIter; i++) {
    const bool alloc = rand_int(0, 2);
    if (alloc || ptrStack.size() == 0) {
      const int size = rand_int(0, (1<<20)); // 0-1 MB allocations
      ptrStack.push(pa.allocate(size));
      sizeStack.push(size);
    }
    else {
      pa.deallocate(ptrStack.top());
      ptrStack.pop();
      sizeStack.pop();
    }
  }

  // TEST: allocation size
  int allocSize = 0;
  while (sizeStack.size() > 0) { allocSize += sizeStack.top(); sizeStack.pop(); }
  std::cout << allocSize << " " << pa.allocatedSize() << std::endl;
  assert( allocSize == pa.allocatedSize() );

  // TEST: number of allocationg
  const int numAllocs = ptrStack.size();
  assert( numAllocs == pa.numUsedBlocks() );

  return 0;
}

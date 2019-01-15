#include "FixedSizePool.hpp"
#include "StdAllocator.hpp"
#include "AllocatorTest.hpp"

#include <cassert>
#include <cstdlib>
#include <stack>

typedef int ValueType;
typedef FixedSizePool<ValueType, StdAllocator, StdAllocator, (1<<1)> PoolType;

std::size_t rand_int(const std::size_t min, const std::size_t max) {
  return min + static_cast<float>(std::rand()) / RAND_MAX * (max - min);
}

int main() {
  std::srand(0);

  PoolType &pa = PoolType::getInstance();
  std::stack<ValueType*> ptrStack;

  const int numIter = (1 << 20);
  for (int i = 0; i < numIter; i++) {
    const bool alloc = rand_int(0, 2);
    if (alloc || ptrStack.size() == 0) {
      ptrStack.push(pa.allocate());
    }
    else {
      pa.deallocate(ptrStack.top());
      ptrStack.pop();
    }
  }

  // TEST: allocation size
  assert( ptrStack.size()*sizeof(ValueType) == pa.allocatedSize() );

  return 0;
}

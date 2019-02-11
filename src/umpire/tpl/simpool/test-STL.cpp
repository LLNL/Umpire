#include "DynamicSizePool.hpp"
#include "StdAllocator.hpp"
#include "AllocatorTest.hpp"

#include <list>

typedef void* PtrType;
typedef std::list< PtrType, STLAllocator<PtrType> > ListType;
typedef DynamicSizePool<AllocatorType> PoolType;

std::size_t rand_int(const std::size_t min, const std::size_t max) {
  return min + static_cast<float>(std::rand()) / RAND_MAX * (max - min);
}

int main() {
  std::srand(0);

  ListType L;
  for (int i = 0; i < 1000; i++) {
    L.push_back(PoolType::getInstance().allocate(rand_int(0, (1<<20))));
  }
  for (PtrType &v : L) PoolType::getInstance().deallocate(v);

  return 0;
}

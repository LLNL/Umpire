#include "AllocatorTest.hpp"

#include <list>

typedef float ValueType;
typedef std::list<ValueType*> ListType;
typedef DynamicSizePool<AllocatorType> PoolType;

std::size_t rand_int(const std::size_t min, const std::size_t max) {
  return min + static_cast<float>(std::rand()) / RAND_MAX * (max - min);
}

int main() {
  std::srand(0);

  ListType L;
  for (int i = 0; i < 1000; i++) L.push_back(new ValueType());
  for (ValueType *v : L) delete v;

  return 0;
}

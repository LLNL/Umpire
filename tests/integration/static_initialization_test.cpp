#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPool.hpp"

static auto& rm{umpire::ResourceManager::getInstance()};

static auto alloc{rm.getAllocator("HOST")};
static auto pool_alloc{rm.makeAllocator<umpire::strategy::DynamicPool>("host_pool", alloc)};

static void* data_1{alloc.allocate(512)};
static void* data_2{pool_alloc.allocate(512)};

int main()
{
  alloc.deallocate(data_1);
  pool_alloc.deallocate(data_2);
  return 0;
}

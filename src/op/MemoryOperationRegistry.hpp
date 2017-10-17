#ifndef UMPIRE_OperationRegistry_HPP
#define UMPIRE_OperationRegistry_HPP

#include "umpire/op/MemoryOperation.hpp"

#include "umpire/AllocatorInterface.hpp"
#include "umpire/util/Platform.hpp"

#include <memory>
#include <unordered_map>
#include <functional>

namespace umpire {
namespace op {

struct pair_hash {
  std::size_t operator () (const std::pair<Platform, Platform> &p) const {
      auto h1 = std::hash<int>{}(static_cast<int>(p.first));
      auto h2 = std::hash<int>{}(static_cast<int>(p.second));

      // Mainly for demonstration purposes, i.e. works but is overly simple
      // In the real world, use sth. like boost.hash_combine
      return h1 ^ h2;
  }
};

class MemoryOperationRegistry {
  public:

    static MemoryOperationRegistry& getInstance();

    std::shared_ptr<umpire::op::MemoryOperation> find(
        const std::string& name,
        std::shared_ptr<AllocatorInterface>& source_allocator,
        std::shared_ptr<AllocatorInterface>& dst_allocator);

    void registerOperation(
      const std::string& name,
      std::pair<Platform, Platform> platforms,
      std::shared_ptr<MemoryOperation>&& operation);

  protected:
    MemoryOperationRegistry();

  private:
    static MemoryOperationRegistry* s_memory_operation_registry_instance;

    std::unordered_map<
      std::string,
      std::unordered_map< std::pair<Platform, Platform>, 
                          std::shared_ptr<MemoryOperation>, 
                          pair_hash > > m_operators;

};

} // end of namespace op
} // end of namespace umpire

#endif

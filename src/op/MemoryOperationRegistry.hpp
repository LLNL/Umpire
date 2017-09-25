#ifndef UMPIRE_OperationRegistry_HPP
#define UMPIRE_OperationRegistry_HPP

namespace umpire {
namespace op {

class MemoryOperationRegistry {
  public:

    static MemoryOperationRegistry& getInstance();

    std::shared_ptr<umpire::op::MemoryOperation> find(
        const std::string& name,
        std::shared_ptr<AllocatorInterface>& source_allocator,
        std::shared_ptr<AllocatorInterface>& dst_allocator);

  protected:
    MemoryOperationRegistry();

  private:
    static MemoryOperationRegistry* s_memory_operation_registry_instance;

    std::unordered_map<
      std::string,
      std::unordered_map< std::pair<Platform, Platform>, 
                          std::shared_ptr<MemoryOperation> > > m_operators;

};

} // end of namespace op
} // end of namespace umpire

#endif

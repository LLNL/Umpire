// splicer begin class.ResourceManager.C_declarations

void umpire_resourcemanager_register_allocation(umpire_resourcemanager * self,
    void* ptr, size_t size, umpire_allocator allocator);

void umpire_resourcemanager_deregister_allocation(umpire_resourcemanager * self, void* ptr);

// splicer end class.ResourceManager.C_declarations

// splicer begin class.ResourceManager.C_definitions

void umpire_resourcemanager_register_allocation(umpire_resourcemanager * self,
    void* ptr, size_t size, umpire_allocator allocator)
{
  umpire::ResourceManager *SH_this =
      static_cast<umpire::ResourceManager *>(self->addr);

  umpire::Allocator *SHCXX_allocator = static_cast<umpire::Allocator *>(allocator.addr);
  umpire::strategy::AllocationStrategy *SHCXX_strategy = SHCXX_allocator->getAllocationStrategy();

  SH_this->registerAllocation(ptr, umpire::util::AllocationRecord{ptr, size, SHCXX_strategy});
}

void umpire_resourcemanager_deregister_allocation(umpire_resourcemanager * self, void* ptr)
{
  umpire::ResourceManager *SH_this =
      static_cast<umpire::ResourceManager *>(self->addr);
  SH_this->deregisterAllocation(ptr);
}

// splicer end class.ResourceManager.C_definitions
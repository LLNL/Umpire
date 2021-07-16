#include "umpire/umpire.hpp"

#include "umpire/detail/io.hpp"
#include "umpire/detail/log.hpp"

volatile int umpire_ver_2_found;

namespace umpire {

void initialize() {
  const char* env_enable_replay{getenv("UMPIRE_REPLAY")};
  const bool enable_replay{env_enable_replay != nullptr};

  const char* env_enable_log{getenv("UMPIRE_LOG_LEVEL")};
  const bool enable_log{env_enable_log != nullptr};

  detail::initialize_io(enable_log, enable_replay);

  register_strategy("HOST", std::unique_ptr<resource::host_memory<>>(resource::host_memory<>::get()));
  // auto& allocator_map = detail::registry::get()->get_allocator_name_map();
  // auto& allocator_list = detail::registry::get()->get_allocator_name_map();
  // allocator_map["HOST"] = resource::host_memory<>::get();
  register_strategy("__zero_byte_pool", std::unique_ptr<resource::null_resource<>>(resource::null_resource<>::get());
}

void finalize()
{
}

std::vector<std::string> get_allocator_names()
{
  auto allocators_by_name = detail::registry::get()->get_allocator_name_map();
  std::vector<std::string> names;
  for (const auto& kv : allocators_by_name) {
    names.push_back(kv.first);
  }
  return names;
}

std::vector<int> get_allocator_ids() { 
  auto allocators_by_id = detail::registry::get()->get_allocator_id_map();
  std::vector<int> ids;
  for (const auto& kv : allocators_by_id) {
    ids.push_back(kv.first);
  }
  return ids;
}

bool is_allocator(const std::string& name) {
  auto allocators_by_name = detail::registry::get()->get_allocator_name_map();

  return (allocators_by_name.find(name) != allocators_by_name.end());
}


memory* get_strategy(const std::string& name)
{
  auto& allocators = detail::registry::get()->get_allocator_name_map();
  auto allocator = allocators.find(name);

  if (allocator == allocators.end()) {
    UMPIRE_ERROR("Allocator \"" << name << "\" not found. Available allocators: ");
        //<< getAllocatorInformation());
  }

  return allocator->second;
}

void register_allocator(const std::string& name, Allocator allocator)
{
  if (is_allocator(name)) {
    UMPIRE_ERROR("Allocator " << name << " is already registered.");
  } else {
    auto& allocator_map = detail::registry::get()->get_allocator_name_map();
    allocator_map[name] = allocator.get_memory();
  }
}


Allocator get_allocator(const std::string& name) {
  return Allocator(get_strategy(name));
}

bool is_allocator(const std::string& name)
{
  auto& a = detail::registry::get()->get_allocator_name_map();
  return (a.find(name) != a.end());
}

int get_device_count()
{
  int device_count{0};
#if defined(UMPIRE_ENABLE_CUDA)
  ::cudaGetDeviceCount(&device_count);
#elif defined(UMPIRE_ENABLE_HIP)
  hipGetDeviceCount(&device_count);
#endif
  return device_count;
}

void print_allocator_records(Allocator allocator, std::ostream& os)
{
  std::stringstream ss;
  auto& rm = umpire::ResourceManager::getInstance();

  auto strategy = allocator.getAllocationStrategy();

  rm.m_allocations.print([strategy] (const util::AllocationRecord& rec) {
    return rec.strategy == strategy;
  }, ss);

  if (! ss.str().empty() ) {
    os << "Allocations for "
      << allocator.getName()
      << " allocator:" << std::endl
      << ss.str() << std::endl;
  }
}

std::vector<allocation_record> get_allocator_records(Allocator allocator)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto strategy = allocator.getAllocationStrategy();

  std::vector<util::AllocationRecord> recs;
  std::copy_if(rm.m_allocations.begin(), rm.m_allocations.end(),
               std::back_inserter(recs), [strategy] (const util::AllocationRecord& rec) {
                 return rec.strategy == strategy;
               });

  return recs;
}

bool pointer_overlaps(void* left_ptr, void* right_ptr)
{
  auto& rm = umpire::ResourceManager::getInstance();

  try {
    auto left_record = rm.findAllocationRecord(left_ptr);
    auto right_record = rm.findAllocationRecord(right_ptr);

    char* left{reinterpret_cast<char*>(left_record->ptr)};
    char* right{reinterpret_cast<char*>(right_record->ptr)};

    return ((right >= left) 
      && ((left + left_record->size) > right)
      && ((right + right_record->size) > (left + left_record->size)));
  } catch (umpire::util::Exception&) {
    UMPIRE_LOG(Error, "Unknown pointer in pointer_overlaps");
    throw;
  }
}

bool pointer_contains(void* left_ptr, void* right_ptr)
{
  auto& rm = umpire::ResourceManager::getInstance();

  try {
    auto left_record = rm.findAllocationRecord(left_ptr);
    auto right_record = rm.findAllocationRecord(right_ptr);

    char* left{reinterpret_cast<char*>(left_record->ptr)};
    char* right{reinterpret_cast<char*>(right_record->ptr)};

    return ((right >= left) 
      && (left + left_record->size > right)
      && (right + right_record->size <= left + left_record->size));
  } catch (umpire::util::Exception&) {
    UMPIRE_LOG(Error, "Unknown pointer in pointer_contains");
    throw;
  }
}

std::string get_backtrace(void* ptr)
{
#if defined(UMPIRE_ENABLE_BACKTRACE)
  auto& rm = umpire::ResourceManager::getInstance();
  auto record = rm.findAllocationRecord(ptr);
  return umpire::util::backtracer<>::print(record->allocation_backtrace);
#else
  UMPIRE_USE_VAR(ptr);
  return "[Umpire: UMPIRE_BACKTRACE=Off]";
#endif
}


std::size_t get_process_memory_usage()
{
#if defined(_MSC_VER) || defined(__APPLE__)
  return 0;
#else
  std::size_t ignore;
  std::size_t resident;
  std::ifstream statm("/proc/self/statm");
  statm >> ignore >> resident >> ignore;
  statm.close();
  long page_size{::sysconf(_SC_PAGE_SIZE)};
  return std::size_t{resident * page_size};
#endif
}

std::size_t get_device_memory_usage(int device_id)
{
#if defined(UMPIRE_ENABLE_CUDA)
  std::size_t mem_free{0};
  std::size_t mem_tot{0};

  int current_device;
  cudaGetDevice(&current_device);

  cudaSetDevice(device_id);

  cudaMemGetInfo(&mem_free, &mem_tot);

  cudaSetDevice(current_device);

  return std::size_t{mem_tot - mem_free};
#else
  UMPIRE_USE_VAR(device_id);
  return 0;
#endif
}

std::vector<allocation_record>
get_leaked_allocations(Allocator allocator)
{
  return get_allocator_records(allocator);
}

}
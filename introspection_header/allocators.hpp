#include <cuda_runtime.h>

namespace umpire {
namespace util {

struct record {
  void* ptr;
  std::size_t size;
};

} // end of namespace util

struct cuda_allocator_tag {};
struct cuda_um_allocator_tag {};
struct host_allocator_tag {};

template <typename Allocator>
struct header_allocator {};

template<>
struct header_allocator<host_allocator_tag>
{
  void* allocate(std::size_t bytes) {
    std::size_t total_size{bytes+sizeof(util::record)};
    uintptr_t ptr{reinterpret_cast<uintptr_t>(malloc(total_size))};

    auto my_record = reinterpret_cast<util::record*>(ptr);
    void* ret{reinterpret_cast<void*>(ptr+sizeof(util::record))};

    my_record->ptr = ret;
    my_record->size = bytes;

    return ret;
  };

  void deallocate(void* ptr) {
    uintptr_t base = reinterpret_cast<uintptr_t>(ptr) - sizeof(util::record);
    free((void*) base);
  }

  util::record* getRecord(void* ptr) {
    uintptr_t base = reinterpret_cast<uintptr_t>(ptr) - sizeof(util::record);
    auto my_record = reinterpret_cast<util::record*>(base);
    //std::cout << "ptr: " << ptr << " record: {"  << my_record->ptr << ", " << my_record->size << "}" << std::endl;
    return my_record;
  }
};

struct mallocator {
  void* allocate(std::size_t bytes) {
    return ::malloc(bytes);
  }

  void deallocate(void* ptr) {
    ::free(ptr);
  }

  util::record* getRecord(void*) {
    return nullptr;
  }
};

template<>
struct header_allocator<cuda_allocator_tag>
{
  header_allocator()
  {
    cudaMallocHost(&m_record, sizeof(util::record));
  }

  void* allocate(std::size_t bytes) {
    void* base_ptr;
    std::size_t total_size{bytes+sizeof(util::record)};

    ::cudaMalloc(&base_ptr, total_size);
    uintptr_t ptr{reinterpret_cast<uintptr_t>(base_ptr)};
    void* ret{reinterpret_cast<void*>(ptr+sizeof(util::record))};

    m_record->ptr = ret;
    m_record->size = bytes;
    cudaMemcpy(base_ptr, m_record, sizeof(util::record), cudaMemcpyHostToDevice);

    return ret;
  };

  void deallocate(void* ptr) {
    uintptr_t base = reinterpret_cast<uintptr_t>(ptr) - sizeof(util::record);
    ::cudaFree((void*) base);
  }

  util::record* getRecord(void* ptr) {
    uintptr_t base = reinterpret_cast<uintptr_t>(ptr) - sizeof(util::record);
    cudaMemcpy(m_record, (void*)base, sizeof(util::record), cudaMemcpyDeviceToHost);
    return m_record;
  }

  util::record* m_record;
};

template<>
struct header_allocator<cuda_um_allocator_tag>
{

  void* allocate(std::size_t bytes) {
    void* base_ptr;
    std::size_t total_size{bytes+sizeof(util::record)};

    ::cudaMallocManaged(&base_ptr, total_size);
    uintptr_t ptr{reinterpret_cast<uintptr_t>(base_ptr)};
    auto my_record = reinterpret_cast<util::record*>(ptr);
    void* ret{reinterpret_cast<void*>(ptr+sizeof(util::record))};
    my_record->ptr = ret;
    my_record->size = bytes;

    cudaMemPrefetchAsync(base_ptr, total_size, 0);

    return ret;
  };

  void deallocate(void* ptr) {
    uintptr_t base = reinterpret_cast<uintptr_t>(ptr) - sizeof(util::record);
    ::cudaFree((void*) base);
  }

  util::record* getRecord(void* ptr) {
    uintptr_t base = reinterpret_cast<uintptr_t>(ptr) - sizeof(util::record);
    auto my_record = reinterpret_cast<util::record*>(base);
    return my_record;
  }

  util::record* m_record;
};

struct cudamallocator {
  void* allocate(std::size_t bytes) {
    void* ret;
    ::cudaMalloc(&ret, bytes);
    return ret;
  }

  void deallocate(void* ptr) {
    ::cudaFree(ptr);
  }

  util::record* getRecord(void*) {
    return nullptr;
  }
};

struct cudaummallocator {
  void* allocate(std::size_t bytes) {
    void* ret;
    ::cudaMallocManaged(&ret, bytes);
    return ret;
  }

  void deallocate(void* ptr) {
    ::cudaFree(ptr);
  }

  util::record* getRecord(void*) {
    return nullptr;
  }
};

}

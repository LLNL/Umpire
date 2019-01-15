# Simpool
Simpool is a very simple pooled memory allocator that offers recipes
for use in C++ by overloading `::operator new(std::size_t)` and
fulfilling an STL allocator concept.

## Background
The concept behind a pooled memory allocator is to reduce the number
of system calls to allocate memory, and instead takes memory from an
already allocated segment of memory. It can be more efficient if there
are many small allocations or if the allocator function incurs
significant overhead.

> Why do we need another memory pool?

While there are many other existing pool implementations, all the
implementations that I can find have a flaw. This code addresses the
following weaknesses:


1. While others create a usable pool for the memory, they do not for
   keeping track of the blocks, meaning these allocations still incur
   the overhead of the system `malloc` for each block. This seems
   like an oversight.

2. Other implementations do not offer the ability to select the memory
   spaces used for both the memory pool internally and for the memory
   pointers provided by the class.

## Design
This code uses a series of pools to represent the internal and
allocated memory. These pools can be in any memory space reachable
from the thread allocation and deallocation function.

## Fixed Type
The `FixedSizePool<T, MA, NP>` class stores "pools" each of
`NP*sizeof(unsigned int)*8` objects of type `T` in the memory space
with allocator struct `MA`. An example of an allocator struct for the
system `malloc()/free()` methods:

``` c++
struct CPUAllocator
{
  static inline void *allocate(std::size_t size) { return std::malloc(size); }
  static inline void deallocate(void *ptr) { std::free(ptr); }
};
```

The class keeps track of which locations in the pool are unused by
flipping single bits. It can do this because the objects are all the
same size.

This algorithm calls the allocator function once per pool creation, so
is guaranteed to call it no more than every `NP*sizeof(unsigned
int)*8` allocations.

The public non-constructor/destructor methods are:

- `T* allocate()`: returns a pointer to memory for an object `T`
- `void deallocate(T* ptr)`: Tells the pool that `ptr` will no longer
  be used. The behavior is undefined if `ptr` was not returned from
  `allocate()` above.
- `std::size_t allocatedSize() const`: Return the allocated size,
  without internal overhead.
- `std::size_t totalSize() const`: Return the total size of all
  allocations within.
- `std::size_t numPools() const`: Return the number of fixed size
  pools.

## Dynamic Type
The `DynamicSizePool<MA, IA, MINSIZE>` class allocates objects
with the `MA` allocator struct, and internally keeps track of the
blocks using a `FixedMemoryPool` using the `IA` allocator struct. Each
of the blocks allocated with `MA` are at least size `MINSIZE` - smaller
allocations are carved out.

This class largely follows the algorithm used in
[cnmem](https://github.com/NVIDIA/cnmem). This involves splitting
blocks if allocations are smaller than `MINSIZE`, and merging blocks
if possible when memory is marked as no longer used. It is therefore
difficult to determine an upper bound on the number of allocations
made with `IA` and `MA`.

The public non-constructor/destructor methods are:

- `void* allocate(std::size_t size)`: returns a pointer to `size` bytes of memory.
- `void deallocate(T* ptr)`: Tells the pool that `ptr` will no longer
  be used. The behavior is undefined if `ptr` was not returned from
  `allocate(std::size_t)` above.
- `std::size_t allocatedSize() const`: Return the allocated size,
  without internal overhead.
- `std::size_t totalSize() const`: Return the total size of the class
  and all allocations within.
- `std::size_t numFreeBlocks() const`: Return the number of free blocks.
- `std::size_t numUsedBlocks() const`: Return the number of used blocks.

## License
&copy; Copyright 2017 IBM Corporation. MIT License.

## Authors
  - Johann Dahm <<johann.dahm@ibm.com>> Primary contact


## See also
- GNU glibc `obstack`: [info page](https://www.gnu.org/software/libc/manual/html_node/Obstacks.html)
- NVidia `cnmem`: [Github page](https://github.com/NVIDIA/cnmem)
- Google Perftools `tcmalloc`: [website](http://goog-perftools.sourceforge.net/doc/tcmalloc.html)
- `jemalloc`: [website](http://jemalloc.net)

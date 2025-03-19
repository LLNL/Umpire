==========
Allocators
==========

Allocators are the fundamental object used to allocate and deallocate memory
using Umpire.

Allocators provide a unified interface to allocate and free data. 
They also encapsulate all the details of how and where allocations will be made, and can also be used to introspect the memory resource. 
The allocator objects do not return typed allocations, so the pointer returned from the allocate method must be cast to the relevant type.

Below are some of the specific types of Allocators Umpire provides in addition to the standard Umpire allocators.

Typed Allocator
--------------

The Typed Allocator is an allocator for objects of type `T`.

This class is an adaptor that allows using an Allocator to allocate objects of type `T`. 
You can use this class as an allocator for STL containers like `std::vector`.

Device Allocator
----------------

The Device Allocator is an allocator specifically designed to be used on the GPU.

The Device Allocator is created with a `size` parameter that specifies the maximum amount of memory
available for the Device Allocator to use. Hence, the Device Allocator doesn't actually allocate
new memory when the ``allocate`` member function is called. Instead, it increments a pointer
atomically to the relevant data.

Shared Memory Allocator
-----------------------

The Shared Memory Allocator is an allocator that is meant for sharing memory across nodes or sockets.

Umpire has a couple different kinds of Shared Memory Allocators. When the allocator is created, a 
size is specified which becomes the maximum size for the allocator. The Shared Memory Allocators also
require a unique name, though Umpire provides a `NamingShim` feature for more ease of use.


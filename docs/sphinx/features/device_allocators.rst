================
Device Allocators
=================

The DeviceAllocator is designed for memory allocations on the GPU. 
Currently there is only support for CUDA, although HIP support is coming soon.

This should include how it is only supported in CUDA right now and relocateable device code must be supported.

Creating a Device Allocator
--------------------------

To create a DeviceAllocator, users can call the `umpire::makeDeviceAllocator` host function.
This function takes an allocator, the total amount of memory you need to be allocated in bytes, and a name
for the new DeviceAllocator.

.. literalinclude:: ../../../examples/device-allocator.cpp
   :start-after: _sphinx_tag_make_dev_allocator_start
   :end-before: _sphinx_tag_make_dev_allocator_end
   :language: C++

When the DeviceAllocator is created, the `size` parameter that is passed to the `umpire::makeDeviceAllocator`
function is the total memory, in bytes, available to that allocator. Whenever the `allocate` function is
called on the GPU, it is simply atomically incrementing a pointer.

Retrieving a DeviceAllocator Object
-----------------------------------

The `umpire::getDeviceAllocator` function returns the DeviceAllocator object that corresponds to the 
ID or name given. The DeviceAllocator class includes helper methods to query the name or ID of a
DeviceAllocator if unknown. Below is an example of using the ID to "get" the DeviceAllocator object:

.. literalinclude:: ../../../examples/device-allocator.cpp
   :start-after: _sphinx_tag_get_dev_allocator_id_start
   :end-before: _sphinx_tag_get_dev_allocator_id_end
   :language: C++

And next is an example of using the name to "get" the DeviceAllocator object:

.. literalinclude:: ../../../examples/device-allocator.cpp
   :start-after: _sphinx_tag_get_dev_allocator_name_start
   :end-before: _sphinx_tag_get_dev_allocator_name_end
   :language: C++

The function can be called from either the host or the device. Users can now
"get" a DeviceAllocator object from inside a kernel without needing to keep track of passing the
object to the kernel.

Under the hood, `umpire::getDeviceAllocator` uses global arrays which can be accessed by both the host and device.
The global array is indexed by DeviceAllocator ID. Because we are using global arrays on host and device,
the arrays need to be synched after the DeviceAllocators are created, but before any kernels are called.
The synching process is done by calling the `UMPIRE_SET_UP_DEVICE_ALLOCS()` macro. This just ensures that
the host and device global arrays are pointing at the same information. Below is an example:

.. literalinclude:: ../../../examples/device-allocator.cpp
   :start-after: _sphinx_tag_macro_start
   :end-before: _sphinx_tag_macro_end
   :language: C++

.. note::
   In order to use the full capabilities of the DeviceAllocator, relocatable device code must be enabled.

Currently, the memory that has been used with the DeviceAllocator is only freed when the DeviceAllocator
is torn down by the destructor. There will be a `reset()` function implemented soon to change this.

.. literalinclude:: ../../../examples/device-allocator.cpp

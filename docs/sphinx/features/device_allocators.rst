.. _device_allocators:

=================
Device Allocators
=================

The DeviceAllocator is designed for memory allocations on the GPU. 

Creating a Device Allocator
--------------------------

To create a DeviceAllocator, users can call the :class:`umpire::make_device_allocator` host function.
This function takes an allocator, the total amount of memory the DeviceAllocator will have, and a unique name
for the new DeviceAllocator object, as shown below. A maximum of 64 unique DeviceAllocators can be
created at a time.

.. literalinclude:: ../../../examples/device-allocator.cpp
   :start-after: _sphinx_tag_make_dev_allocator_start
   :end-before: _sphinx_tag_make_dev_allocator_end
   :language: C++

When the DeviceAllocator is created, the ``size`` parameter that is passed to the :class:`umpire::make_device_allocator`
function is the total memory, in bytes, available to that allocator. Whenever the ``allocate`` function is
called on the GPU, it is simply atomically incrementing a counter which offsets a pointer to the start of that
memory. In other words, the total size from all of the allocates performed on the device with the DeviceAllocator may not 
exceed the size that was used when creating the device allocator.

To see what the total memory, in bytes, available to the allocator is, simply call the :class:`DeviceAllocator::getTotalSize()`
function.

Retrieving a DeviceAllocator Object
-----------------------------------

After creating a DeviceAllocator, we can immediately start using that allocator to allocate device memory. To do this, we
have the :class:`umpire::get_device_allocator` host/device function which returns the DeviceAllocator object corresponding 
to the name (or ID) given. The DeviceAllocator class also includes a helper function, :class:`umpire::is_device_allocator`,
to query whether or not a given name (or ID) corresponds to an existing DeviceAllocator. Below is an example of using 
the **name** to obtain the DeviceAllocator object:

.. literalinclude:: ../../../examples/device-allocator.cpp
   :start-after: _sphinx_tag_get_dev_allocator_name_start
   :end-before: _sphinx_tag_get_dev_allocator_name_end
   :language: C++

With the :class:`umpire::get_device_allocator` function, there is no need to keep track of a DeviceAllocator, since function
call stacks can become quite complex. Users can instead use this function to obtain it inside whichever host or device
function they need.

.. note::
   When compiling without relocatable device code (RDC), the ``UMPIRE_SET_UP_DEVICE_ALLOCATORS()`` macro must be called 
   in every translation unit that will use the :class:`umpire::get_device_allocator` function.

Resetting Memory on the DeviceAllocator
---------------------------------------

The memory that has been used with the DeviceAllocator is only freed at the end of a program when the 
ResourceManager is torn down. However, there is a way to overwrite old or outdated data. Users can call
the :class:`DeviceAllocator::reset()` method which will allows old data to be overwritten. Below is an example:

.. literalinclude:: ../../../examples/device-allocator.cpp
   :start-after: _sphinx_tag_reset_start
   :end-before: _sphinx_tag_reset_end
   :language: C++

The above code snippet shows the ``reset()`` function being called from the host. Calling the function from the host
utilizes the ResourceManager and Umpire's ``memset`` operation under the hood. Therefore, there is some kind of 
synchronization guaranteed. However, if the ``reset()`` function is called on the device, there is no synchronization
guaranteed, so the user must be very careful not to reset memory that other GPU threads still need.

To see the current size of the DeviceAllocator (aka, the current amount of memory, in bytes, being used), call the
:class:`DeviceAllocator::getCurrentSize()` function.

.. literalinclude:: ../../../examples/device-allocator.cpp

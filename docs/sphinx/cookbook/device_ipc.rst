.. _device_ipc:

===========================
Sharing GPU Memory with IPC
===========================

Umpire provides a ``DeviceIpcAllocator`` strategy that allows GPU memory to be
shared between processes on the same node using CUDA or HIP IPC handles. This
strategy uses the shared memory infrastructure in Umpire to coordinate the
sharing of GPU memory between processes.

How It Works
------------

The ``DeviceIpcAllocator`` works by:

1. Having one process (the "leader", determined by the "scope"-local MPI rank 0) physically allocate GPU memory
2. The leader process gets an IPC handle for that memory and stores it in CPU shared memory
3. Other processes retrieve the IPC handle from CPU shared memory and import the GPU memory
4. MPI barriers are used to synchronize between processes and ensure safe access

This allows multiple processes to share GPU memory efficiently, which is useful
for multi-process applications that need to operate on the same data.

Using DeviceIpcAllocator
------------------------

A ``DeviceIpcAllocator`` can be created using the Umpire ResourceMangager. It
doesn't require any additional arguments apart from a name. By default, device
memory will be allocated using the "DEVICE" resource. Optionally, you can
provide a device allocator, as well as a "scope" argument that determines who
will share each GPU allocation. 

Here's an example:

.. literalinclude:: ../../../examples/cookbook/recipe_device_ipc.cpp
   :language: cpp

The scope argument can either be "socket" or "node". The "socket" scope means
that all processes using the same socket (as identified by the PCI address of
the GPU) will share the GPU memory, while the "node" scope means that all
processes on the same node (as determined by MPI_COMM_TYPE_SHARED) will share
the GPU memory.

Limitations
-----------

- Requires UMPIRE_ENABLE_MPI to be enabled
- Requires UMPIRE_ENABLE_IPC_SHARED_MEMORY to be enabled

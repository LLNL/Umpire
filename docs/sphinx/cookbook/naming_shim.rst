.. _naming_shim:

==========================================
Using a Naming Shim with IPC Shared Memory
==========================================

Using a unique name for IPC Shared Memory allocators can get tricky, especially
when dealing with many MPI tasks and in an integrated code set-up.

The `NamingShim` was created to make it a bit easier to set up and use the IPC
Shared Memory Allocator without having to always worry about the unique name requirement.


##############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
from __future__ import print_function

types = (
    ( 'int', 'integer(C_INT)' ),
    ( 'long', 'integer(C_LONG)' ),
    ( 'float', 'real(C_FLOAT)' ),
    ( 'double', 'real(C_DOUBLE)' )
)

allocators = [ "HOST", "DEVICE", "UM"]

maxdims = 3

def gen_allocate_test_subroutines():
    print('')

    print("""
module umpire_fortran_generated_tests

  use iso_c_binding
  use fruit
  use umpire_mod
  implicit none

  contains
""")

    print('')

    for alloc in allocators:
        if alloc == "DEVICE":
            print('#ifdef UMPIRE_ENABLE_CUDA')

        for (name, c_type) in types:
            for dim in range(maxdims+1):
                print("""
      subroutine test_allocate_{name}_{alloc_lower}_array_{dim}d
        use iso_c_binding

        type(UmpireResourceManager) rm
        type(UmpireAllocator) allocator

        {c_type}, pointer, dimension({dim_string}) :: array

        rm = rm%get_instance()
        allocator = rm%get_allocator_by_name("{alloc}")


        call allocator%allocate(array, [{sizes}])
        call assert_true(associated(array))

        call allocator%deallocate(array)
        call assert_true(.not. associated(array))

        call allocator%delete()
      end subroutine test_allocate_{name}_{alloc_lower}_array_{dim}d

    """.format(
            dim=dim+1, 
            name=name, 
            c_type=c_type, 
            dim_string= ", ".join([":" for i in range(dim+1)]), 
            sizes=", ".join(["10" for i in range(dim+1)]),
            index=", ".join(["0" for i in range(dim+1)]),
            alloc=alloc,
            alloc_lower=alloc.lower()
        ))

        if alloc == "UM":
            print('#endif')

    print("""
end module umpire_fortran_generated_tests
""")


def gen_allocate_test_calls():
    for alloc in allocators:
        if alloc == "DEVICE":
            print('#ifdef UMPIRE_ENABLE_CUDA')

        for (name, c_type) in types:
            for dim in range(maxdims+1):
                print('  call test_allocate_{name}_{alloc}_array_{dim}d'.format(dim=dim+1, name=name, alloc=alloc.lower()))

        if alloc == "UM":
            print('#endif')


def gen_fortran():
    print('! Generated by genumpireftests.py')
    print('')

    gen_allocate_test_subroutines()

    print("""
program fortran_test
  use fruit
  use umpire_fortran_generated_tests

  implicit none
  logical ok

  call init_fruit
""")

    gen_allocate_test_calls()

    print("""
  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
    call exit(1)
  endif
end program fortran_test
""")

if __name__ == '__main__':
    gen_fortran()

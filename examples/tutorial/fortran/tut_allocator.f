!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
! project contributors. See the COPYRIGHT file for details.
!
! SPDX-License-Identifier: (MIT)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
program fortran_test
      use umpire_mod
      implicit none
      logical ok

      integer(C_INT), pointer :: array(:)
      type(UmpireAllocator) allocator
      type(UmpireResourceManager) rm

      ! _umpire_tut_get_allocator_start
      rm = rm%get_instance()
      allocator = rm%get_allocator_by_id(0)
      ! _umpire_tut_get_allocator_end

      ! _umpire_tut_allocate_start
      call allocator%allocate(array, [ 10 ])

      write(10,*) "Allocated array of size ", 10

      call allocator%deallocate(array)
      ! _umpire_tut_allocate_end

      write(10,*) "deallocated."
end program fortran_test

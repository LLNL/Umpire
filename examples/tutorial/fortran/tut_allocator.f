!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
! Produced at the Lawrence Livermore National Laboratory
!
! Created by David Beckingsale, david@llnl.gov
! LLNL-CODE-747640
!
! All rights reserved.
!
! This file is part of Umpire.
!
! For details, see https://github.com/LLNL/Umpire
! Please also see the LICENSE file for MIT license.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
program fortran_test
      use umpire_mod
      implicit none
      logical ok

      integer(C_INT), pointer :: array(:)
      type(UmpireAllocator) allocator
      type(UmpireResourceManager) rm

      rm = rm%get_instance()
      allocator = rm%get_allocator_by_id(0)

      call allocator%allocate(array, [ 10 ])

      write(10,*) "Allocated array of size ", 10

      call allocator%deallocate(array)

      write(10,*) "deallocated."
end program fortran_test

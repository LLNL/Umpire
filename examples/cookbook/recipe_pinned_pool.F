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
    
program umpire_f_pinned_pool
      use umpire_mod
      implicit none
      logical ok

      integer(C_INT), pointer, dimension(:) :: array(:)
      type(UmpireAllocator) base_allocator
      type(UmpireAllocator) pinned_pool
      type(UmpireResourceManager) rm

      rm = rm%get_instance()
      base_allocator = rm%get_allocator_by_name("PINNED")

      pinned_pool = rm%make_allocator_pool("PINNED_POOL", &
                                           base_allocator, &
                                           512*1024, &
                                           1024)

      call pinned_pool%allocate(array, [10])
end program umpire_f_pinned_pool

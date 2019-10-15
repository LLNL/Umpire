!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
! project contributors. See the COPYRIGHT file for details.
!
! SPDX-License-Identifier: (MIT)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
program umpire_f_pinned_pool
      use umpire_mod
      implicit none
      logical ok

      integer(C_INT), pointer, dimension(:) :: array(:)
      type(UmpireAllocator) base_allocator
      type(UmpireAllocator) prefetch_pool
      type(UmpireResourceManager) rm

      rm = rm%get_instance()
      base_allocator = rm%get_allocator_by_name("UM")

      prefetch_allocator = rm%make_allocator_pool("prefetcher", &
                                           base_allocator, 0)

      call prefetch_allocator%allocate(array, [10])
end program umpire_f_pinned_pool

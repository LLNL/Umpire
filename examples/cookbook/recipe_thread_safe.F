!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
! project contributors. See the COPYRIGHT file for details.
!
! SPDX-License-Identifier: (MIT)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

program umpire_f_pinned_pool
      use umpire_mod
      implicit none
      logical ok

      integer(C_INT), pointer, dimension(:) :: array(:)
      type(UmpireAllocator) base_allocator
      type(UmpireAllocator) pool
      type(UmpireAllocator) thread_safe_allocator
      type(UmpireResourceManager) rm

      rm = rm%get_instance()
      base_allocator = rm%get_allocator_by_name("HOST")

      pool = rm%make_allocator_pool("POOL",         &
                                    base_allocator, &
                                    512_8*1024_8,   &
                                    1024_8)

      thread_safe_allocator = rm%make_allocator_thread_safe("THREAD_SAFE_POOL", &
                                                            pool)

      call thread_safe_allocator%allocate(array, [10])
end program umpire_f_pinned_pool

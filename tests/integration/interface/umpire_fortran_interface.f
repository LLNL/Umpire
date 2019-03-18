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

module umpire_fortran_test
  use iso_c_binding
  use fruit
  use umpire_mod
  implicit none

contains
  subroutine allocate_sixty_four
    integer(C_INT), pointer :: array(:)
    type(C_PTR) data_ptr
    integer i

    type(UmpireAllocator) allocator
    type(UmpireResourceManager) rm

    rm = rm%get_instance()

    allocator = rm%get_allocator_by_id(0)

    data_ptr = allocator%allocate(c_sizeof(i)*10);
    call c_f_pointer(data_ptr, array, [ 10 ])

    do i = 1, 10
      array(i) = i*i
    enddo

    write(*, *) (array(i), i=1,10)

  end subroutine allocate_sixty_four

end module umpire_fortran_test

program fortran_test
  use fruit
  use umpire_fortran_test
  implicit none
  logical ok

  call init_fruit

  call allocate_sixty_four

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
    call exit(1)
  endif
end program fortran_test

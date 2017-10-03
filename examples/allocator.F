program main

  use iso_c_binding
  use umpire_mod
  implicit none

  type(UmpireResourceManager) rm
  type(UmpireAllocator) allocator

  integer(4), pointer :: my_allocation(:)

  integer(8) :: size
  integer :: i

  rm = resourcemanager_get()

  allocator = rm%get_allocator("HOST")

  print *, "Allocating 100 integers..."
  size = 100

  my_allocation = allocator%allocate_int(size)

  do i=0,100
    my_allocation(i) = 0.1
  enddo


  print *, "my_allocation(", 10, ") = ", my_allocation(10)

end program main

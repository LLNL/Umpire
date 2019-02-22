! splicer begin class.Allocator.additional_functions
subroutine allocator_allocate_int_array_1d(this, dims, array)
      type(UmpireAllocator) :: this

      integer(C_INT), intent(inout), pointer, dimension(:) :: array
      type(C_PTR) :: data_ptr
      type(C_INT) :: size_type

      use iso_c_binding

      data_ptr = allocator%allocate(product(dims) * sizeof(size_type))
      call c_f_pointer(dataptr, array, dims)
end subroutine allocator_allocate_int_array_1d
! splicer end class.Allocator.additional_functions

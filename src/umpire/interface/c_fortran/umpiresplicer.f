! splicer begin class.Allocator.type_bound_procedure_part
procedure :: allocate_int_array_1d => allocator_allocate_int_array_1d
generic, public :: allocate_array => allocate_int_array_1d
! splicer end class.Allocator.type_bound_procedure_part


! splicer begin class.Allocator.additional_functions
subroutine allocator_allocate_int_array_1d(this, dims, array)
      use iso_c_binding

      class(UmpireAllocator) :: this
      integer(C_INT), intent(inout), pointer, dimension(:) :: array

      integer, dimension(:) :: dims

      type(C_PTR) :: data_ptr

      integer(C_INT) :: size_type

      data_ptr = allocator%allocate(product(dims) * sizeof(size_type))
      call c_f_pointer(data_ptr, array, dims)
end subroutine allocator_allocate_int_array_1d
! splicer end class.Allocator.additional_functions

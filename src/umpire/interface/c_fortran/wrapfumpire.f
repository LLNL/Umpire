! wrapfumpire.f
! This is generated code, do not edit
! Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
! project contributors. See the COPYRIGHT file for details.
!
! SPDX-License-Identifier: (MIT)
!>
!! \file wrapfumpire.f
!! \brief Shroud generated wrapper for umpire namespace
!<
! splicer begin file_top
! splicer end file_top
module umpire_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR, C_SIZE_T
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    type, bind(C) :: SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_capsule_data

    type, bind(C) :: SHROUD_array
        type(SHROUD_capsule_data) :: cxx       ! address of C++ memory
        type(C_PTR) :: addr = C_NULL_PTR       ! address of data in cxx
        integer(C_SIZE_T) :: len = 0_C_SIZE_T  ! bytes-per-item or character len of data in cxx
        integer(C_SIZE_T) :: size = 0_C_SIZE_T ! size of data in cxx
    end type SHROUD_array

    type, bind(C) :: SHROUD_allocator_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_allocator_capsule

    type UmpireAllocator
        type(SHROUD_allocator_capsule) :: cxxmem
        ! splicer begin class.Allocator.component_part
        ! splicer end class.Allocator.component_part
    contains
        procedure :: delete => allocator_delete
        procedure :: allocate_pointer => allocator_allocate
        procedure :: deallocate_pointer => allocator_deallocate
        procedure :: release => allocator_release
        procedure :: get_size => allocator_get_size
        procedure :: get_high_watermark => allocator_get_high_watermark
        procedure :: get_current_size => allocator_get_current_size
        procedure :: get_actual_size => allocator_get_actual_size
        procedure :: get_name => allocator_get_name
        procedure :: get_id => allocator_get_id
        procedure :: get_instance => allocator_get_instance
        procedure :: set_instance => allocator_set_instance
        procedure :: associated => allocator_associated
        ! splicer begin class.Allocator.type_bound_procedure_part

        procedure :: allocate_int_array_1d => allocator_allocate_int_array_1d
        procedure :: deallocate_int_array_1d => allocator_deallocate_int_array_1d
        procedure :: allocate_int_array_2d => allocator_allocate_int_array_2d
        procedure :: deallocate_int_array_2d => allocator_deallocate_int_array_2d
        procedure :: allocate_int_array_3d => allocator_allocate_int_array_3d
        procedure :: deallocate_int_array_3d => allocator_deallocate_int_array_3d
        procedure :: allocate_int_array_4d => allocator_allocate_int_array_4d
        procedure :: deallocate_int_array_4d => allocator_deallocate_int_array_4d
        procedure :: allocate_long_array_1d => allocator_allocate_long_array_1d
        procedure :: deallocate_long_array_1d => allocator_deallocate_long_array_1d
        procedure :: allocate_long_array_2d => allocator_allocate_long_array_2d
        procedure :: deallocate_long_array_2d => allocator_deallocate_long_array_2d
        procedure :: allocate_long_array_3d => allocator_allocate_long_array_3d
        procedure :: deallocate_long_array_3d => allocator_deallocate_long_array_3d
        procedure :: allocate_long_array_4d => allocator_allocate_long_array_4d
        procedure :: deallocate_long_array_4d => allocator_deallocate_long_array_4d
        procedure :: allocate_float_array_1d => allocator_allocate_float_array_1d
        procedure :: deallocate_float_array_1d => allocator_deallocate_float_array_1d
        procedure :: allocate_float_array_2d => allocator_allocate_float_array_2d
        procedure :: deallocate_float_array_2d => allocator_deallocate_float_array_2d
        procedure :: allocate_float_array_3d => allocator_allocate_float_array_3d
        procedure :: deallocate_float_array_3d => allocator_deallocate_float_array_3d
        procedure :: allocate_float_array_4d => allocator_allocate_float_array_4d
        procedure :: deallocate_float_array_4d => allocator_deallocate_float_array_4d
        procedure :: allocate_double_array_1d => allocator_allocate_double_array_1d
        procedure :: deallocate_double_array_1d => allocator_deallocate_double_array_1d
        procedure :: allocate_double_array_2d => allocator_allocate_double_array_2d
        procedure :: deallocate_double_array_2d => allocator_deallocate_double_array_2d
        procedure :: allocate_double_array_3d => allocator_allocate_double_array_3d
        procedure :: deallocate_double_array_3d => allocator_deallocate_double_array_3d
        procedure :: allocate_double_array_4d => allocator_allocate_double_array_4d
        procedure :: deallocate_double_array_4d => allocator_deallocate_double_array_4d
        generic, public :: allocate => &
            allocate_int_array_1d, &
            allocate_int_array_2d, &
            allocate_int_array_3d, &
            allocate_int_array_4d, &
            allocate_long_array_1d, &
            allocate_long_array_2d, &
            allocate_long_array_3d, &
            allocate_long_array_4d, &
            allocate_float_array_1d, &
            allocate_float_array_2d, &
            allocate_float_array_3d, &
            allocate_float_array_4d, &
            allocate_double_array_1d, &
            allocate_double_array_2d, &
            allocate_double_array_3d, &
            allocate_double_array_4d

        generic, public :: deallocate => &
            deallocate_int_array_1d, &
            deallocate_int_array_2d, &
            deallocate_int_array_3d, &
            deallocate_int_array_4d, &
            deallocate_long_array_1d, &
            deallocate_long_array_2d, &
            deallocate_long_array_3d, &
            deallocate_long_array_4d, &
            deallocate_float_array_1d, &
            deallocate_float_array_2d, &
            deallocate_float_array_3d, &
            deallocate_float_array_4d, &
            deallocate_double_array_1d, &
            deallocate_double_array_2d, &
            deallocate_double_array_3d, &
            deallocate_double_array_4d

        ! splicer end class.Allocator.type_bound_procedure_part
    end type UmpireAllocator

    type, bind(C) :: SHROUD_resourcemanager_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_resourcemanager_capsule

    type UmpireResourceManager
        type(SHROUD_resourcemanager_capsule) :: cxxmem
        ! splicer begin class.ResourceManager.component_part
        ! splicer end class.ResourceManager.component_part
    contains
        procedure, nopass :: get_instance => resourcemanager_get_instance
        procedure :: get_allocator_by_name => resourcemanager_get_allocator_by_name
        procedure :: get_allocator_by_id => resourcemanager_get_allocator_by_id
        procedure :: make_allocator_pool => resourcemanager_make_allocator_pool
        procedure :: make_allocator_list_pool => resourcemanager_make_allocator_list_pool
        procedure :: make_allocator_advisor => resourcemanager_make_allocator_advisor
        procedure :: make_allocator_named => resourcemanager_make_allocator_named
        procedure :: make_allocator_fixed_pool => resourcemanager_make_allocator_fixed_pool
        procedure :: register_allocator => resourcemanager_register_allocator
        procedure :: get_allocator_for_ptr => resourcemanager_get_allocator_for_ptr
        procedure :: is_allocator => resourcemanager_is_allocator
        procedure :: has_allocator => resourcemanager_has_allocator
        procedure :: copy_all => resourcemanager_copy_all
        procedure :: copy_with_size => resourcemanager_copy_with_size
        procedure :: memset_all => resourcemanager_memset_all
        procedure :: memset_with_size => resourcemanager_memset_with_size
        procedure :: reallocate_default => resourcemanager_reallocate_default
        procedure :: reallocate_with_allocator => resourcemanager_reallocate_with_allocator
        procedure :: move => resourcemanager_move
        procedure :: deallocate => resourcemanager_deallocate
        procedure :: get_size => resourcemanager_get_size
        procedure :: associated => resourcemanager_associated
        generic :: copy => copy_all, copy_with_size
        generic :: get_allocator => get_allocator_by_name,  &
            get_allocator_by_id, get_allocator_for_ptr
        generic :: memset => memset_all, memset_with_size
        generic :: reallocate => reallocate_default,  &
            reallocate_with_allocator
        ! splicer begin class.ResourceManager.type_bound_procedure_part
        ! splicer end class.ResourceManager.type_bound_procedure_part
    end type UmpireResourceManager

    interface operator (.eq.)
        module procedure allocator_eq
        module procedure resourcemanager_eq
    end interface

    interface operator (.ne.)
        module procedure allocator_ne
        module procedure resourcemanager_ne
    end interface

    interface

        subroutine c_allocator_delete(self) &
                bind(C, name="umpire_allocator_delete")
            import :: SHROUD_allocator_capsule
            implicit none
            type(SHROUD_allocator_capsule), intent(IN) :: self
        end subroutine c_allocator_delete

        function c_allocator_allocate(self, bytes) &
                result(SHT_rv) &
                bind(C, name="umpire_allocator_allocate")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            import :: SHROUD_allocator_capsule
            implicit none
            type(SHROUD_allocator_capsule), intent(IN) :: self
            integer(C_SIZE_T), value, intent(IN) :: bytes
            type(C_PTR) :: SHT_rv
        end function c_allocator_allocate

        subroutine c_allocator_deallocate(self, ptr) &
                bind(C, name="umpire_allocator_deallocate")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_allocator_capsule
            implicit none
            type(SHROUD_allocator_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
        end subroutine c_allocator_deallocate

        subroutine c_allocator_release(self) &
                bind(C, name="umpire_allocator_release")
            import :: SHROUD_allocator_capsule
            implicit none
            type(SHROUD_allocator_capsule), intent(IN) :: self
        end subroutine c_allocator_release

        function c_allocator_get_size(self, ptr) &
                result(SHT_rv) &
                bind(C, name="umpire_allocator_get_size")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            import :: SHROUD_allocator_capsule
            implicit none
            type(SHROUD_allocator_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
            integer(C_SIZE_T) :: SHT_rv
        end function c_allocator_get_size

        function c_allocator_get_high_watermark(self) &
                result(SHT_rv) &
                bind(C, name="umpire_allocator_get_high_watermark")
            use iso_c_binding, only : C_SIZE_T
            import :: SHROUD_allocator_capsule
            implicit none
            type(SHROUD_allocator_capsule), intent(IN) :: self
            integer(C_SIZE_T) :: SHT_rv
        end function c_allocator_get_high_watermark

        function c_allocator_get_current_size(self) &
                result(SHT_rv) &
                bind(C, name="umpire_allocator_get_current_size")
            use iso_c_binding, only : C_SIZE_T
            import :: SHROUD_allocator_capsule
            implicit none
            type(SHROUD_allocator_capsule), intent(IN) :: self
            integer(C_SIZE_T) :: SHT_rv
        end function c_allocator_get_current_size

        function c_allocator_get_actual_size(self) &
                result(SHT_rv) &
                bind(C, name="umpire_allocator_get_actual_size")
            use iso_c_binding, only : C_SIZE_T
            import :: SHROUD_allocator_capsule
            implicit none
            type(SHROUD_allocator_capsule), intent(IN) :: self
            integer(C_SIZE_T) :: SHT_rv
        end function c_allocator_get_actual_size

        pure function c_allocator_get_name(self) &
                result(SHT_rv) &
                bind(C, name="umpire_allocator_get_name")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_allocator_capsule
            implicit none
            type(SHROUD_allocator_capsule), intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_allocator_get_name

        subroutine c_allocator_get_name_bufferify(self, DSHF_rv) &
                bind(C, name="umpire_allocator_get_name_bufferify")
            import :: SHROUD_allocator_capsule, SHROUD_array
            implicit none
            type(SHROUD_allocator_capsule), intent(IN) :: self
            type(SHROUD_array), intent(INOUT) :: DSHF_rv
        end subroutine c_allocator_get_name_bufferify

        function c_allocator_get_id(self) &
                result(SHT_rv) &
                bind(C, name="umpire_allocator_get_id")
            use iso_c_binding, only : C_SIZE_T
            import :: SHROUD_allocator_capsule
            implicit none
            type(SHROUD_allocator_capsule), intent(IN) :: self
            integer(C_SIZE_T) :: SHT_rv
        end function c_allocator_get_id

        ! splicer begin class.Allocator.additional_interfaces
        ! splicer end class.Allocator.additional_interfaces

        function c_resourcemanager_get_instance(SHT_crv) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_get_instance")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_resourcemanager_get_instance

        function c_resourcemanager_get_allocator_by_name(self, name, &
                SHT_crv) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_get_allocator_by_name")
            use iso_c_binding, only : C_CHAR, C_PTR
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            type(SHROUD_allocator_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_resourcemanager_get_allocator_by_name

        function c_resourcemanager_get_allocator_by_name_bufferify(self, &
                name, Lname, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_get_allocator_by_name_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
            type(SHROUD_allocator_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_resourcemanager_get_allocator_by_name_bufferify

        function c_resourcemanager_get_allocator_by_id(self, id, &
                SHT_crv) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_get_allocator_by_id")
            use iso_c_binding, only : C_INT, C_PTR
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: id
            type(SHROUD_allocator_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_resourcemanager_get_allocator_by_id

        function c_resourcemanager_make_allocator_pool(self, name, &
                allocator, initial_size, block, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_make_allocator_pool")
            use iso_c_binding, only : C_CHAR, C_PTR, C_SIZE_T
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            type(SHROUD_allocator_capsule), value, intent(IN) :: allocator
            integer(C_SIZE_T), value, intent(IN) :: initial_size
            integer(C_SIZE_T), value, intent(IN) :: block
            type(SHROUD_allocator_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_resourcemanager_make_allocator_pool

        function c_resourcemanager_make_allocator_bufferify_pool(self, &
                name, Lname, allocator, initial_size, block, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_make_allocator_bufferify_pool")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR, C_SIZE_T
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
            type(SHROUD_allocator_capsule), value, intent(IN) :: allocator
            integer(C_SIZE_T), value, intent(IN) :: initial_size
            integer(C_SIZE_T), value, intent(IN) :: block
            type(SHROUD_allocator_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_resourcemanager_make_allocator_bufferify_pool

        function c_resourcemanager_make_allocator_list_pool(self, name, &
                allocator, initial_size, block, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_make_allocator_list_pool")
            use iso_c_binding, only : C_CHAR, C_PTR, C_SIZE_T
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            type(SHROUD_allocator_capsule), value, intent(IN) :: allocator
            integer(C_SIZE_T), value, intent(IN) :: initial_size
            integer(C_SIZE_T), value, intent(IN) :: block
            type(SHROUD_allocator_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_resourcemanager_make_allocator_list_pool

        function c_resourcemanager_make_allocator_bufferify_list_pool( &
                self, name, Lname, allocator, initial_size, block, &
                SHT_crv) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_make_allocator_bufferify_list_pool")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR, C_SIZE_T
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
            type(SHROUD_allocator_capsule), value, intent(IN) :: allocator
            integer(C_SIZE_T), value, intent(IN) :: initial_size
            integer(C_SIZE_T), value, intent(IN) :: block
            type(SHROUD_allocator_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_resourcemanager_make_allocator_bufferify_list_pool

        function c_resourcemanager_make_allocator_advisor(self, name, &
                allocator, advice_op, device_id, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_make_allocator_advisor")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            type(SHROUD_allocator_capsule), value, intent(IN) :: allocator
            character(kind=C_CHAR), intent(IN) :: advice_op(*)
            integer(C_INT), value, intent(IN) :: device_id
            type(SHROUD_allocator_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_resourcemanager_make_allocator_advisor

        function c_resourcemanager_make_allocator_bufferify_advisor( &
                self, name, Lname, allocator, advice_op, Ladvice_op, &
                device_id, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_make_allocator_bufferify_advisor")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
            type(SHROUD_allocator_capsule), value, intent(IN) :: allocator
            character(kind=C_CHAR), intent(IN) :: advice_op(*)
            integer(C_INT), value, intent(IN) :: Ladvice_op
            integer(C_INT), value, intent(IN) :: device_id
            type(SHROUD_allocator_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_resourcemanager_make_allocator_bufferify_advisor

        function c_resourcemanager_make_allocator_named(self, name, &
                allocator, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_make_allocator_named")
            use iso_c_binding, only : C_CHAR, C_PTR
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            type(SHROUD_allocator_capsule), value, intent(IN) :: allocator
            type(SHROUD_allocator_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_resourcemanager_make_allocator_named

        function c_resourcemanager_make_allocator_bufferify_named(self, &
                name, Lname, allocator, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_make_allocator_bufferify_named")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
            type(SHROUD_allocator_capsule), value, intent(IN) :: allocator
            type(SHROUD_allocator_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_resourcemanager_make_allocator_bufferify_named

        function c_resourcemanager_make_allocator_fixed_pool(self, name, &
                allocator, object_size, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_make_allocator_fixed_pool")
            use iso_c_binding, only : C_CHAR, C_PTR, C_SIZE_T
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            type(SHROUD_allocator_capsule), value, intent(IN) :: allocator
            integer(C_SIZE_T), value, intent(IN) :: object_size
            type(SHROUD_allocator_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_resourcemanager_make_allocator_fixed_pool

        function c_resourcemanager_make_allocator_bufferify_fixed_pool( &
                self, name, Lname, allocator, object_size, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_make_allocator_bufferify_fixed_pool")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR, C_SIZE_T
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
            type(SHROUD_allocator_capsule), value, intent(IN) :: allocator
            integer(C_SIZE_T), value, intent(IN) :: object_size
            type(SHROUD_allocator_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_resourcemanager_make_allocator_bufferify_fixed_pool

        subroutine c_resourcemanager_register_allocator(self, name, &
                allocator) &
                bind(C, name="umpire_resourcemanager_register_allocator")
            use iso_c_binding, only : C_CHAR
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            type(SHROUD_allocator_capsule), value, intent(IN) :: allocator
        end subroutine c_resourcemanager_register_allocator

        subroutine c_resourcemanager_register_allocator_bufferify(self, &
                name, Lname, allocator) &
                bind(C, name="umpire_resourcemanager_register_allocator_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
            type(SHROUD_allocator_capsule), value, intent(IN) :: allocator
        end subroutine c_resourcemanager_register_allocator_bufferify

        function c_resourcemanager_get_allocator_for_ptr(self, ptr, &
                SHT_crv) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_get_allocator_for_ptr")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
            type(SHROUD_allocator_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_resourcemanager_get_allocator_for_ptr

        function c_resourcemanager_is_allocator(self, name) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_is_allocator")
            use iso_c_binding, only : C_BOOL, C_CHAR
            import :: SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            logical(C_BOOL) :: SHT_rv
        end function c_resourcemanager_is_allocator

        function c_resourcemanager_is_allocator_bufferify(self, name, &
                Lname) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_is_allocator_bufferify")
            use iso_c_binding, only : C_BOOL, C_CHAR, C_INT
            import :: SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
            logical(C_BOOL) :: SHT_rv
        end function c_resourcemanager_is_allocator_bufferify

        function c_resourcemanager_has_allocator(self, ptr) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_has_allocator")
            use iso_c_binding, only : C_BOOL, C_PTR
            import :: SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
            logical(C_BOOL) :: SHT_rv
        end function c_resourcemanager_has_allocator

        subroutine c_resourcemanager_copy_all(self, src_ptr, dst_ptr) &
                bind(C, name="umpire_resourcemanager_copy_all")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: src_ptr
            type(C_PTR), value, intent(IN) :: dst_ptr
        end subroutine c_resourcemanager_copy_all

        subroutine c_resourcemanager_copy_with_size(self, src_ptr, &
                dst_ptr, size) &
                bind(C, name="umpire_resourcemanager_copy_with_size")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            import :: SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: src_ptr
            type(C_PTR), value, intent(IN) :: dst_ptr
            integer(C_SIZE_T), value, intent(IN) :: size
        end subroutine c_resourcemanager_copy_with_size

        subroutine c_resourcemanager_memset_all(self, ptr, val) &
                bind(C, name="umpire_resourcemanager_memset_all")
            use iso_c_binding, only : C_INT, C_PTR
            import :: SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
            integer(C_INT), value, intent(IN) :: val
        end subroutine c_resourcemanager_memset_all

        subroutine c_resourcemanager_memset_with_size(self, ptr, val, &
                length) &
                bind(C, name="umpire_resourcemanager_memset_with_size")
            use iso_c_binding, only : C_INT, C_PTR, C_SIZE_T
            import :: SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
            integer(C_INT), value, intent(IN) :: val
            integer(C_SIZE_T), value, intent(IN) :: length
        end subroutine c_resourcemanager_memset_with_size

        function c_resourcemanager_reallocate_default(self, src_ptr, &
                size) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_reallocate_default")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            import :: SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: src_ptr
            integer(C_SIZE_T), value, intent(IN) :: size
            type(C_PTR) :: SHT_rv
        end function c_resourcemanager_reallocate_default

        function c_resourcemanager_reallocate_with_allocator(self, &
                src_ptr, size, allocator) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_reallocate_with_allocator")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: src_ptr
            integer(C_SIZE_T), value, intent(IN) :: size
            type(SHROUD_allocator_capsule), value, intent(IN) :: allocator
            type(C_PTR) :: SHT_rv
        end function c_resourcemanager_reallocate_with_allocator

        function c_resourcemanager_move(self, src_ptr, allocator) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_move")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: src_ptr
            type(SHROUD_allocator_capsule), value, intent(IN) :: allocator
            type(C_PTR) :: SHT_rv
        end function c_resourcemanager_move

        subroutine c_resourcemanager_deallocate(self, ptr) &
                bind(C, name="umpire_resourcemanager_deallocate")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
        end subroutine c_resourcemanager_deallocate

        function c_resourcemanager_get_size(self, ptr) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_get_size")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            import :: SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
            integer(C_SIZE_T) :: SHT_rv
        end function c_resourcemanager_get_size

        ! splicer begin class.ResourceManager.additional_interfaces
        ! splicer end class.ResourceManager.additional_interfaces

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

    interface
        ! helper function
        ! Copy the char* or std::string in context into c_var.
        subroutine SHROUD_copy_string_and_free(context, c_var, c_var_size) &
             bind(c,name="umpire_ShroudCopyStringAndFree")
            use, intrinsic :: iso_c_binding, only : C_CHAR, C_SIZE_T
            import SHROUD_array
            type(SHROUD_array), intent(IN) :: context
            character(kind=C_CHAR), intent(OUT) :: c_var(*)
            integer(C_SIZE_T), value :: c_var_size
        end subroutine SHROUD_copy_string_and_free
    end interface

contains

    subroutine allocator_delete(obj)
        class(UmpireAllocator) :: obj
        ! splicer begin class.Allocator.method.delete
        call c_allocator_delete(obj%cxxmem)
        ! splicer end class.Allocator.method.delete
    end subroutine allocator_delete

    function allocator_allocate(obj, bytes) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR, C_SIZE_T
        class(UmpireAllocator) :: obj
        integer(C_SIZE_T), value, intent(IN) :: bytes
        type(C_PTR) :: SHT_rv
        ! splicer begin class.Allocator.method.allocate_pointer
        SHT_rv = c_allocator_allocate(obj%cxxmem, bytes)
        ! splicer end class.Allocator.method.allocate_pointer
    end function allocator_allocate

    subroutine allocator_deallocate(obj, ptr)
        use iso_c_binding, only : C_PTR
        class(UmpireAllocator) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        ! splicer begin class.Allocator.method.deallocate_pointer
        call c_allocator_deallocate(obj%cxxmem, ptr)
        ! splicer end class.Allocator.method.deallocate_pointer
    end subroutine allocator_deallocate

    subroutine allocator_release(obj)
        class(UmpireAllocator) :: obj
        ! splicer begin class.Allocator.method.release
        call c_allocator_release(obj%cxxmem)
        ! splicer end class.Allocator.method.release
    end subroutine allocator_release

    function allocator_get_size(obj, ptr) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR, C_SIZE_T
        class(UmpireAllocator) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        integer(C_SIZE_T) :: SHT_rv
        ! splicer begin class.Allocator.method.get_size
        SHT_rv = c_allocator_get_size(obj%cxxmem, ptr)
        ! splicer end class.Allocator.method.get_size
    end function allocator_get_size

    function allocator_get_high_watermark(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_SIZE_T
        class(UmpireAllocator) :: obj
        integer(C_SIZE_T) :: SHT_rv
        ! splicer begin class.Allocator.method.get_high_watermark
        SHT_rv = c_allocator_get_high_watermark(obj%cxxmem)
        ! splicer end class.Allocator.method.get_high_watermark
    end function allocator_get_high_watermark

    function allocator_get_current_size(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_SIZE_T
        class(UmpireAllocator) :: obj
        integer(C_SIZE_T) :: SHT_rv
        ! splicer begin class.Allocator.method.get_current_size
        SHT_rv = c_allocator_get_current_size(obj%cxxmem)
        ! splicer end class.Allocator.method.get_current_size
    end function allocator_get_current_size

    function allocator_get_actual_size(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_SIZE_T
        class(UmpireAllocator) :: obj
        integer(C_SIZE_T) :: SHT_rv
        ! splicer begin class.Allocator.method.get_actual_size
        SHT_rv = c_allocator_get_actual_size(obj%cxxmem)
        ! splicer end class.Allocator.method.get_actual_size
    end function allocator_get_actual_size

    function allocator_get_name(obj) &
            result(SHT_rv)
        class(UmpireAllocator) :: obj
        type(SHROUD_array) :: DSHF_rv
        character(len=:), allocatable :: SHT_rv
        ! splicer begin class.Allocator.method.get_name
        call c_allocator_get_name_bufferify(obj%cxxmem, DSHF_rv)
        ! splicer end class.Allocator.method.get_name
        allocate(character(len=DSHF_rv%len):: SHT_rv)
        call SHROUD_copy_string_and_free(DSHF_rv, SHT_rv, DSHF_rv%len)
    end function allocator_get_name

    function allocator_get_id(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_SIZE_T
        class(UmpireAllocator) :: obj
        integer(C_SIZE_T) :: SHT_rv
        ! splicer begin class.Allocator.method.get_id
        SHT_rv = c_allocator_get_id(obj%cxxmem)
        ! splicer end class.Allocator.method.get_id
    end function allocator_get_id

    ! Return pointer to C++ memory.
    function allocator_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(UmpireAllocator), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function allocator_get_instance

    subroutine allocator_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(UmpireAllocator), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine allocator_set_instance

    function allocator_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(UmpireAllocator), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function allocator_associated

    ! splicer begin class.Allocator.additional_functions


    subroutine allocator_allocate_int_array_1d(this, array, dims)
          use iso_c_binding

          class(UmpireAllocator) :: this
          integer(C_INT), intent(inout), pointer, dimension(:) :: array

          integer, dimension(:) :: dims

          type(C_PTR) :: data_ptr

          integer(C_INT) :: size_type
          integer(C_SIZE_T) :: num_bytes

          num_bytes = product(dims) * sizeof(size_type)
          data_ptr = this%allocate_pointer(num_bytes)

          call c_f_pointer(data_ptr, array, dims)
    end subroutine allocator_allocate_int_array_1d



    subroutine allocator_deallocate_int_array_1d(this, array)
          use iso_c_binding

          class(UmpireAllocator) :: this
          integer(C_INT), intent(inout), pointer, dimension(:) :: array

          type(C_PTR) :: data_ptr

          data_ptr = c_loc(array)

          call this%deallocate_pointer(data_ptr)
          nullify(array)
    end subroutine allocator_deallocate_int_array_1d



    subroutine allocator_allocate_int_array_2d(this, array, dims)
          use iso_c_binding

          class(UmpireAllocator) :: this
          integer(C_INT), intent(inout), pointer, dimension(:, :) :: array

          integer, dimension(:) :: dims

          type(C_PTR) :: data_ptr

          integer(C_INT) :: size_type
          integer(C_SIZE_T) :: num_bytes

          num_bytes = product(dims) * sizeof(size_type)
          data_ptr = this%allocate_pointer(num_bytes)

          call c_f_pointer(data_ptr, array, dims)
    end subroutine allocator_allocate_int_array_2d



    subroutine allocator_deallocate_int_array_2d(this, array)
          use iso_c_binding

          class(UmpireAllocator) :: this
          integer(C_INT), intent(inout), pointer, dimension(:, :) :: array

          type(C_PTR) :: data_ptr

          data_ptr = c_loc(array)

          call this%deallocate_pointer(data_ptr)
          nullify(array)
    end subroutine allocator_deallocate_int_array_2d



    subroutine allocator_allocate_int_array_3d(this, array, dims)
          use iso_c_binding

          class(UmpireAllocator) :: this
          integer(C_INT), intent(inout), pointer, dimension(:, :, :) :: array

          integer, dimension(:) :: dims

          type(C_PTR) :: data_ptr

          integer(C_INT) :: size_type
          integer(C_SIZE_T) :: num_bytes

          num_bytes = product(dims) * sizeof(size_type)
          data_ptr = this%allocate_pointer(num_bytes)

          call c_f_pointer(data_ptr, array, dims)
    end subroutine allocator_allocate_int_array_3d



    subroutine allocator_deallocate_int_array_3d(this, array)
          use iso_c_binding

          class(UmpireAllocator) :: this
          integer(C_INT), intent(inout), pointer, dimension(:, :, :) :: array

          type(C_PTR) :: data_ptr

          data_ptr = c_loc(array)

          call this%deallocate_pointer(data_ptr)
          nullify(array)
    end subroutine allocator_deallocate_int_array_3d



    subroutine allocator_allocate_int_array_4d(this, array, dims)
          use iso_c_binding

          class(UmpireAllocator) :: this
          integer(C_INT), intent(inout), pointer, dimension(:, :, :, :) :: array

          integer, dimension(:) :: dims

          type(C_PTR) :: data_ptr

          integer(C_INT) :: size_type
          integer(C_SIZE_T) :: num_bytes

          num_bytes = product(dims) * sizeof(size_type)
          data_ptr = this%allocate_pointer(num_bytes)

          call c_f_pointer(data_ptr, array, dims)
    end subroutine allocator_allocate_int_array_4d



    subroutine allocator_deallocate_int_array_4d(this, array)
          use iso_c_binding

          class(UmpireAllocator) :: this
          integer(C_INT), intent(inout), pointer, dimension(:, :, :, :) :: array

          type(C_PTR) :: data_ptr

          data_ptr = c_loc(array)

          call this%deallocate_pointer(data_ptr)
          nullify(array)
    end subroutine allocator_deallocate_int_array_4d



    subroutine allocator_allocate_long_array_1d(this, array, dims)
          use iso_c_binding

          class(UmpireAllocator) :: this
          integer(C_LONG), intent(inout), pointer, dimension(:) :: array

          integer, dimension(:) :: dims

          type(C_PTR) :: data_ptr

          integer(C_LONG) :: size_type
          integer(C_SIZE_T) :: num_bytes

          num_bytes = product(dims) * sizeof(size_type)
          data_ptr = this%allocate_pointer(num_bytes)

          call c_f_pointer(data_ptr, array, dims)
    end subroutine allocator_allocate_long_array_1d



    subroutine allocator_deallocate_long_array_1d(this, array)
          use iso_c_binding

          class(UmpireAllocator) :: this
          integer(C_LONG), intent(inout), pointer, dimension(:) :: array

          type(C_PTR) :: data_ptr

          data_ptr = c_loc(array)

          call this%deallocate_pointer(data_ptr)
          nullify(array)
    end subroutine allocator_deallocate_long_array_1d



    subroutine allocator_allocate_long_array_2d(this, array, dims)
          use iso_c_binding

          class(UmpireAllocator) :: this
          integer(C_LONG), intent(inout), pointer, dimension(:, :) :: array

          integer, dimension(:) :: dims

          type(C_PTR) :: data_ptr

          integer(C_LONG) :: size_type
          integer(C_SIZE_T) :: num_bytes

          num_bytes = product(dims) * sizeof(size_type)
          data_ptr = this%allocate_pointer(num_bytes)

          call c_f_pointer(data_ptr, array, dims)
    end subroutine allocator_allocate_long_array_2d



    subroutine allocator_deallocate_long_array_2d(this, array)
          use iso_c_binding

          class(UmpireAllocator) :: this
          integer(C_LONG), intent(inout), pointer, dimension(:, :) :: array

          type(C_PTR) :: data_ptr

          data_ptr = c_loc(array)

          call this%deallocate_pointer(data_ptr)
          nullify(array)
    end subroutine allocator_deallocate_long_array_2d



    subroutine allocator_allocate_long_array_3d(this, array, dims)
          use iso_c_binding

          class(UmpireAllocator) :: this
          integer(C_LONG), intent(inout), pointer, dimension(:, :, :) :: array

          integer, dimension(:) :: dims

          type(C_PTR) :: data_ptr

          integer(C_LONG) :: size_type
          integer(C_SIZE_T) :: num_bytes

          num_bytes = product(dims) * sizeof(size_type)
          data_ptr = this%allocate_pointer(num_bytes)

          call c_f_pointer(data_ptr, array, dims)
    end subroutine allocator_allocate_long_array_3d



    subroutine allocator_deallocate_long_array_3d(this, array)
          use iso_c_binding

          class(UmpireAllocator) :: this
          integer(C_LONG), intent(inout), pointer, dimension(:, :, :) :: array

          type(C_PTR) :: data_ptr

          data_ptr = c_loc(array)

          call this%deallocate_pointer(data_ptr)
          nullify(array)
    end subroutine allocator_deallocate_long_array_3d



    subroutine allocator_allocate_long_array_4d(this, array, dims)
          use iso_c_binding

          class(UmpireAllocator) :: this
          integer(C_LONG), intent(inout), pointer, dimension(:, :, :, :) :: array

          integer, dimension(:) :: dims

          type(C_PTR) :: data_ptr

          integer(C_LONG) :: size_type
          integer(C_SIZE_T) :: num_bytes

          num_bytes = product(dims) * sizeof(size_type)
          data_ptr = this%allocate_pointer(num_bytes)

          call c_f_pointer(data_ptr, array, dims)
    end subroutine allocator_allocate_long_array_4d



    subroutine allocator_deallocate_long_array_4d(this, array)
          use iso_c_binding

          class(UmpireAllocator) :: this
          integer(C_LONG), intent(inout), pointer, dimension(:, :, :, :) :: array

          type(C_PTR) :: data_ptr

          data_ptr = c_loc(array)

          call this%deallocate_pointer(data_ptr)
          nullify(array)
    end subroutine allocator_deallocate_long_array_4d



    subroutine allocator_allocate_float_array_1d(this, array, dims)
          use iso_c_binding

          class(UmpireAllocator) :: this
          real(C_FLOAT), intent(inout), pointer, dimension(:) :: array

          integer, dimension(:) :: dims

          type(C_PTR) :: data_ptr

          real(C_FLOAT) :: size_type
          integer(C_SIZE_T) :: num_bytes

          num_bytes = product(dims) * sizeof(size_type)
          data_ptr = this%allocate_pointer(num_bytes)

          call c_f_pointer(data_ptr, array, dims)
    end subroutine allocator_allocate_float_array_1d



    subroutine allocator_deallocate_float_array_1d(this, array)
          use iso_c_binding

          class(UmpireAllocator) :: this
          real(C_FLOAT), intent(inout), pointer, dimension(:) :: array

          type(C_PTR) :: data_ptr

          data_ptr = c_loc(array)

          call this%deallocate_pointer(data_ptr)
          nullify(array)
    end subroutine allocator_deallocate_float_array_1d



    subroutine allocator_allocate_float_array_2d(this, array, dims)
          use iso_c_binding

          class(UmpireAllocator) :: this
          real(C_FLOAT), intent(inout), pointer, dimension(:, :) :: array

          integer, dimension(:) :: dims

          type(C_PTR) :: data_ptr

          real(C_FLOAT) :: size_type
          integer(C_SIZE_T) :: num_bytes

          num_bytes = product(dims) * sizeof(size_type)
          data_ptr = this%allocate_pointer(num_bytes)

          call c_f_pointer(data_ptr, array, dims)
    end subroutine allocator_allocate_float_array_2d



    subroutine allocator_deallocate_float_array_2d(this, array)
          use iso_c_binding

          class(UmpireAllocator) :: this
          real(C_FLOAT), intent(inout), pointer, dimension(:, :) :: array

          type(C_PTR) :: data_ptr

          data_ptr = c_loc(array)

          call this%deallocate_pointer(data_ptr)
          nullify(array)
    end subroutine allocator_deallocate_float_array_2d



    subroutine allocator_allocate_float_array_3d(this, array, dims)
          use iso_c_binding

          class(UmpireAllocator) :: this
          real(C_FLOAT), intent(inout), pointer, dimension(:, :, :) :: array

          integer, dimension(:) :: dims

          type(C_PTR) :: data_ptr

          real(C_FLOAT) :: size_type
          integer(C_SIZE_T) :: num_bytes

          num_bytes = product(dims) * sizeof(size_type)
          data_ptr = this%allocate_pointer(num_bytes)

          call c_f_pointer(data_ptr, array, dims)
    end subroutine allocator_allocate_float_array_3d



    subroutine allocator_deallocate_float_array_3d(this, array)
          use iso_c_binding

          class(UmpireAllocator) :: this
          real(C_FLOAT), intent(inout), pointer, dimension(:, :, :) :: array

          type(C_PTR) :: data_ptr

          data_ptr = c_loc(array)

          call this%deallocate_pointer(data_ptr)
          nullify(array)
    end subroutine allocator_deallocate_float_array_3d



    subroutine allocator_allocate_float_array_4d(this, array, dims)
          use iso_c_binding

          class(UmpireAllocator) :: this
          real(C_FLOAT), intent(inout), pointer, dimension(:, :, :, :) :: array

          integer, dimension(:) :: dims

          type(C_PTR) :: data_ptr

          real(C_FLOAT) :: size_type
          integer(C_SIZE_T) :: num_bytes

          num_bytes = product(dims) * sizeof(size_type)
          data_ptr = this%allocate_pointer(num_bytes)

          call c_f_pointer(data_ptr, array, dims)
    end subroutine allocator_allocate_float_array_4d



    subroutine allocator_deallocate_float_array_4d(this, array)
          use iso_c_binding

          class(UmpireAllocator) :: this
          real(C_FLOAT), intent(inout), pointer, dimension(:, :, :, :) :: array

          type(C_PTR) :: data_ptr

          data_ptr = c_loc(array)

          call this%deallocate_pointer(data_ptr)
          nullify(array)
    end subroutine allocator_deallocate_float_array_4d



    subroutine allocator_allocate_double_array_1d(this, array, dims)
          use iso_c_binding

          class(UmpireAllocator) :: this
          real(C_DOUBLE), intent(inout), pointer, dimension(:) :: array

          integer, dimension(:) :: dims

          type(C_PTR) :: data_ptr

          real(C_DOUBLE) :: size_type
          integer(C_SIZE_T) :: num_bytes

          num_bytes = product(dims) * sizeof(size_type)
          data_ptr = this%allocate_pointer(num_bytes)

          call c_f_pointer(data_ptr, array, dims)
    end subroutine allocator_allocate_double_array_1d



    subroutine allocator_deallocate_double_array_1d(this, array)
          use iso_c_binding

          class(UmpireAllocator) :: this
          real(C_DOUBLE), intent(inout), pointer, dimension(:) :: array

          type(C_PTR) :: data_ptr

          data_ptr = c_loc(array)

          call this%deallocate_pointer(data_ptr)
          nullify(array)
    end subroutine allocator_deallocate_double_array_1d



    subroutine allocator_allocate_double_array_2d(this, array, dims)
          use iso_c_binding

          class(UmpireAllocator) :: this
          real(C_DOUBLE), intent(inout), pointer, dimension(:, :) :: array

          integer, dimension(:) :: dims

          type(C_PTR) :: data_ptr

          real(C_DOUBLE) :: size_type
          integer(C_SIZE_T) :: num_bytes

          num_bytes = product(dims) * sizeof(size_type)
          data_ptr = this%allocate_pointer(num_bytes)

          call c_f_pointer(data_ptr, array, dims)
    end subroutine allocator_allocate_double_array_2d



    subroutine allocator_deallocate_double_array_2d(this, array)
          use iso_c_binding

          class(UmpireAllocator) :: this
          real(C_DOUBLE), intent(inout), pointer, dimension(:, :) :: array

          type(C_PTR) :: data_ptr

          data_ptr = c_loc(array)

          call this%deallocate_pointer(data_ptr)
          nullify(array)
    end subroutine allocator_deallocate_double_array_2d



    subroutine allocator_allocate_double_array_3d(this, array, dims)
          use iso_c_binding

          class(UmpireAllocator) :: this
          real(C_DOUBLE), intent(inout), pointer, dimension(:, :, :) :: array

          integer, dimension(:) :: dims

          type(C_PTR) :: data_ptr

          real(C_DOUBLE) :: size_type
          integer(C_SIZE_T) :: num_bytes

          num_bytes = product(dims) * sizeof(size_type)
          data_ptr = this%allocate_pointer(num_bytes)

          call c_f_pointer(data_ptr, array, dims)
    end subroutine allocator_allocate_double_array_3d



    subroutine allocator_deallocate_double_array_3d(this, array)
          use iso_c_binding

          class(UmpireAllocator) :: this
          real(C_DOUBLE), intent(inout), pointer, dimension(:, :, :) :: array

          type(C_PTR) :: data_ptr

          data_ptr = c_loc(array)

          call this%deallocate_pointer(data_ptr)
          nullify(array)
    end subroutine allocator_deallocate_double_array_3d



    subroutine allocator_allocate_double_array_4d(this, array, dims)
          use iso_c_binding

          class(UmpireAllocator) :: this
          real(C_DOUBLE), intent(inout), pointer, dimension(:, :, :, :) :: array

          integer, dimension(:) :: dims

          type(C_PTR) :: data_ptr

          real(C_DOUBLE) :: size_type
          integer(C_SIZE_T) :: num_bytes

          num_bytes = product(dims) * sizeof(size_type)
          data_ptr = this%allocate_pointer(num_bytes)

          call c_f_pointer(data_ptr, array, dims)
    end subroutine allocator_allocate_double_array_4d



    subroutine allocator_deallocate_double_array_4d(this, array)
          use iso_c_binding

          class(UmpireAllocator) :: this
          real(C_DOUBLE), intent(inout), pointer, dimension(:, :, :, :) :: array

          type(C_PTR) :: data_ptr

          data_ptr = c_loc(array)

          call this%deallocate_pointer(data_ptr)
          nullify(array)
    end subroutine allocator_deallocate_double_array_4d


    ! splicer end class.Allocator.additional_functions

    function resourcemanager_get_instance() &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        type(C_PTR) :: SHT_prv
        type(UmpireResourceManager) :: SHT_rv
        ! splicer begin class.ResourceManager.method.get_instance
        SHT_prv = c_resourcemanager_get_instance(SHT_rv%cxxmem)
        ! splicer end class.ResourceManager.method.get_instance
    end function resourcemanager_get_instance

    function resourcemanager_get_allocator_by_name(obj, name) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR
        class(UmpireResourceManager) :: obj
        character(len=*), intent(IN) :: name
        type(C_PTR) :: SHT_prv
        type(UmpireAllocator) :: SHT_rv
        ! splicer begin class.ResourceManager.method.get_allocator_by_name
        SHT_prv = c_resourcemanager_get_allocator_by_name_bufferify(obj%cxxmem, &
            name, len_trim(name, kind=C_INT), SHT_rv%cxxmem)
        ! splicer end class.ResourceManager.method.get_allocator_by_name
    end function resourcemanager_get_allocator_by_name

    function resourcemanager_get_allocator_by_id(obj, id) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR
        class(UmpireResourceManager) :: obj
        integer(C_INT), value, intent(IN) :: id
        type(C_PTR) :: SHT_prv
        type(UmpireAllocator) :: SHT_rv
        ! splicer begin class.ResourceManager.method.get_allocator_by_id
        SHT_prv = c_resourcemanager_get_allocator_by_id(obj%cxxmem, id, &
            SHT_rv%cxxmem)
        ! splicer end class.ResourceManager.method.get_allocator_by_id
    end function resourcemanager_get_allocator_by_id

    function resourcemanager_make_allocator_pool(obj, name, allocator, &
            initial_size, block) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR, C_SIZE_T
        class(UmpireResourceManager) :: obj
        character(len=*), intent(IN) :: name
        type(UmpireAllocator), value, intent(IN) :: allocator
        integer(C_SIZE_T), value, intent(IN) :: initial_size
        integer(C_SIZE_T), value, intent(IN) :: block
        type(C_PTR) :: SHT_prv
        type(UmpireAllocator) :: SHT_rv
        ! splicer begin class.ResourceManager.method.make_allocator_pool
        SHT_prv = c_resourcemanager_make_allocator_bufferify_pool(obj%cxxmem, &
            name, len_trim(name, kind=C_INT), allocator%cxxmem, &
            initial_size, block, SHT_rv%cxxmem)
        ! splicer end class.ResourceManager.method.make_allocator_pool
    end function resourcemanager_make_allocator_pool

    function resourcemanager_make_allocator_list_pool(obj, name, &
            allocator, initial_size, block) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR, C_SIZE_T
        class(UmpireResourceManager) :: obj
        character(len=*), intent(IN) :: name
        type(UmpireAllocator), value, intent(IN) :: allocator
        integer(C_SIZE_T), value, intent(IN) :: initial_size
        integer(C_SIZE_T), value, intent(IN) :: block
        type(C_PTR) :: SHT_prv
        type(UmpireAllocator) :: SHT_rv
        ! splicer begin class.ResourceManager.method.make_allocator_list_pool
        SHT_prv = c_resourcemanager_make_allocator_bufferify_list_pool(obj%cxxmem, &
            name, len_trim(name, kind=C_INT), allocator%cxxmem, &
            initial_size, block, SHT_rv%cxxmem)
        ! splicer end class.ResourceManager.method.make_allocator_list_pool
    end function resourcemanager_make_allocator_list_pool

    function resourcemanager_make_allocator_advisor(obj, name, &
            allocator, advice_op, device_id) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR
        class(UmpireResourceManager) :: obj
        character(len=*), intent(IN) :: name
        type(UmpireAllocator), value, intent(IN) :: allocator
        character(len=*), intent(IN) :: advice_op
        integer(C_INT), value, intent(IN) :: device_id
        type(C_PTR) :: SHT_prv
        type(UmpireAllocator) :: SHT_rv
        ! splicer begin class.ResourceManager.method.make_allocator_advisor
        SHT_prv = c_resourcemanager_make_allocator_bufferify_advisor(obj%cxxmem, &
            name, len_trim(name, kind=C_INT), allocator%cxxmem, &
            advice_op, len_trim(advice_op, kind=C_INT), device_id, &
            SHT_rv%cxxmem)
        ! splicer end class.ResourceManager.method.make_allocator_advisor
    end function resourcemanager_make_allocator_advisor

    function resourcemanager_make_allocator_named(obj, name, allocator) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR
        class(UmpireResourceManager) :: obj
        character(len=*), intent(IN) :: name
        type(UmpireAllocator), value, intent(IN) :: allocator
        type(C_PTR) :: SHT_prv
        type(UmpireAllocator) :: SHT_rv
        ! splicer begin class.ResourceManager.method.make_allocator_named
        SHT_prv = c_resourcemanager_make_allocator_bufferify_named(obj%cxxmem, &
            name, len_trim(name, kind=C_INT), allocator%cxxmem, &
            SHT_rv%cxxmem)
        ! splicer end class.ResourceManager.method.make_allocator_named
    end function resourcemanager_make_allocator_named

    function resourcemanager_make_allocator_fixed_pool(obj, name, &
            allocator, object_size) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR, C_SIZE_T
        class(UmpireResourceManager) :: obj
        character(len=*), intent(IN) :: name
        type(UmpireAllocator), value, intent(IN) :: allocator
        integer(C_SIZE_T), value, intent(IN) :: object_size
        type(C_PTR) :: SHT_prv
        type(UmpireAllocator) :: SHT_rv
        ! splicer begin class.ResourceManager.method.make_allocator_fixed_pool
        SHT_prv = c_resourcemanager_make_allocator_bufferify_fixed_pool(obj%cxxmem, &
            name, len_trim(name, kind=C_INT), allocator%cxxmem, &
            object_size, SHT_rv%cxxmem)
        ! splicer end class.ResourceManager.method.make_allocator_fixed_pool
    end function resourcemanager_make_allocator_fixed_pool

    subroutine resourcemanager_register_allocator(obj, name, allocator)
        use iso_c_binding, only : C_INT
        class(UmpireResourceManager) :: obj
        character(len=*), intent(IN) :: name
        type(UmpireAllocator), value, intent(IN) :: allocator
        ! splicer begin class.ResourceManager.method.register_allocator
        call c_resourcemanager_register_allocator_bufferify(obj%cxxmem, &
            name, len_trim(name, kind=C_INT), allocator%cxxmem)
        ! splicer end class.ResourceManager.method.register_allocator
    end subroutine resourcemanager_register_allocator

    function resourcemanager_get_allocator_for_ptr(obj, ptr) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        type(C_PTR) :: SHT_prv
        type(UmpireAllocator) :: SHT_rv
        ! splicer begin class.ResourceManager.method.get_allocator_for_ptr
        SHT_prv = c_resourcemanager_get_allocator_for_ptr(obj%cxxmem, &
            ptr, SHT_rv%cxxmem)
        ! splicer end class.ResourceManager.method.get_allocator_for_ptr
    end function resourcemanager_get_allocator_for_ptr

    function resourcemanager_is_allocator(obj, name) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL, C_INT
        class(UmpireResourceManager) :: obj
        character(len=*), intent(IN) :: name
        logical :: SHT_rv
        ! splicer begin class.ResourceManager.method.is_allocator
        SHT_rv = c_resourcemanager_is_allocator_bufferify(obj%cxxmem, &
            name, len_trim(name, kind=C_INT))
        ! splicer end class.ResourceManager.method.is_allocator
    end function resourcemanager_is_allocator

    function resourcemanager_has_allocator(obj, ptr) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL, C_PTR
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        logical :: SHT_rv
        ! splicer begin class.ResourceManager.method.has_allocator
        SHT_rv = c_resourcemanager_has_allocator(obj%cxxmem, ptr)
        ! splicer end class.ResourceManager.method.has_allocator
    end function resourcemanager_has_allocator

    subroutine resourcemanager_copy_all(obj, src_ptr, dst_ptr)
        use iso_c_binding, only : C_PTR
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: src_ptr
        type(C_PTR), value, intent(IN) :: dst_ptr
        ! splicer begin class.ResourceManager.method.copy_all
        call c_resourcemanager_copy_all(obj%cxxmem, src_ptr, dst_ptr)
        ! splicer end class.ResourceManager.method.copy_all
    end subroutine resourcemanager_copy_all

    subroutine resourcemanager_copy_with_size(obj, src_ptr, dst_ptr, &
            size)
        use iso_c_binding, only : C_PTR, C_SIZE_T
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: src_ptr
        type(C_PTR), value, intent(IN) :: dst_ptr
        integer(C_SIZE_T), value, intent(IN) :: size
        ! splicer begin class.ResourceManager.method.copy_with_size
        call c_resourcemanager_copy_with_size(obj%cxxmem, src_ptr, &
            dst_ptr, size)
        ! splicer end class.ResourceManager.method.copy_with_size
    end subroutine resourcemanager_copy_with_size

    subroutine resourcemanager_memset_all(obj, ptr, val)
        use iso_c_binding, only : C_INT, C_PTR
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        integer(C_INT), value, intent(IN) :: val
        ! splicer begin class.ResourceManager.method.memset_all
        call c_resourcemanager_memset_all(obj%cxxmem, ptr, val)
        ! splicer end class.ResourceManager.method.memset_all
    end subroutine resourcemanager_memset_all

    subroutine resourcemanager_memset_with_size(obj, ptr, val, length)
        use iso_c_binding, only : C_INT, C_PTR, C_SIZE_T
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        integer(C_INT), value, intent(IN) :: val
        integer(C_SIZE_T), value, intent(IN) :: length
        ! splicer begin class.ResourceManager.method.memset_with_size
        call c_resourcemanager_memset_with_size(obj%cxxmem, ptr, val, &
            length)
        ! splicer end class.ResourceManager.method.memset_with_size
    end subroutine resourcemanager_memset_with_size

    function resourcemanager_reallocate_default(obj, src_ptr, size) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR, C_SIZE_T
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: src_ptr
        integer(C_SIZE_T), value, intent(IN) :: size
        type(C_PTR) :: SHT_rv
        ! splicer begin class.ResourceManager.method.reallocate_default
        SHT_rv = c_resourcemanager_reallocate_default(obj%cxxmem, &
            src_ptr, size)
        ! splicer end class.ResourceManager.method.reallocate_default
    end function resourcemanager_reallocate_default

    function resourcemanager_reallocate_with_allocator(obj, src_ptr, &
            size, allocator) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR, C_SIZE_T
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: src_ptr
        integer(C_SIZE_T), value, intent(IN) :: size
        type(UmpireAllocator), value, intent(IN) :: allocator
        type(C_PTR) :: SHT_rv
        ! splicer begin class.ResourceManager.method.reallocate_with_allocator
        SHT_rv = c_resourcemanager_reallocate_with_allocator(obj%cxxmem, &
            src_ptr, size, allocator%cxxmem)
        ! splicer end class.ResourceManager.method.reallocate_with_allocator
    end function resourcemanager_reallocate_with_allocator

    function resourcemanager_move(obj, src_ptr, allocator) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: src_ptr
        type(UmpireAllocator), value, intent(IN) :: allocator
        type(C_PTR) :: SHT_rv
        ! splicer begin class.ResourceManager.method.move
        SHT_rv = c_resourcemanager_move(obj%cxxmem, src_ptr, &
            allocator%cxxmem)
        ! splicer end class.ResourceManager.method.move
    end function resourcemanager_move

    subroutine resourcemanager_deallocate(obj, ptr)
        use iso_c_binding, only : C_PTR
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        ! splicer begin class.ResourceManager.method.deallocate
        call c_resourcemanager_deallocate(obj%cxxmem, ptr)
        ! splicer end class.ResourceManager.method.deallocate
    end subroutine resourcemanager_deallocate

    function resourcemanager_get_size(obj, ptr) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR, C_SIZE_T
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        integer(C_SIZE_T) :: SHT_rv
        ! splicer begin class.ResourceManager.method.get_size
        SHT_rv = c_resourcemanager_get_size(obj%cxxmem, ptr)
        ! splicer end class.ResourceManager.method.get_size
    end function resourcemanager_get_size

    function resourcemanager_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(UmpireResourceManager), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function resourcemanager_associated

    ! splicer begin class.ResourceManager.additional_functions
    ! splicer end class.ResourceManager.additional_functions

    ! splicer begin additional_functions
    ! splicer end additional_functions

    function allocator_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(UmpireAllocator), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function allocator_eq

    function allocator_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(UmpireAllocator), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function allocator_ne

    function resourcemanager_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(UmpireResourceManager), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function resourcemanager_eq

    function resourcemanager_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(UmpireResourceManager), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function resourcemanager_ne

end module umpire_mod

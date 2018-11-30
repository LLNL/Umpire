! wrapfumpire.f
! This is generated code, do not edit
! Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
!>
!! \file wrapfumpire.f
!! \brief Shroud generated wrapper for Umpire library
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

    ! splicer begin class.DynamicPool.module_top
    ! splicer end class.DynamicPool.module_top

    type, bind(C) :: SHROUD_dynamicpool_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_dynamicpool_capsule

    type dynamicpool
        type(SHROUD_dynamicpool_capsule) :: cxxmem
        ! splicer begin class.DynamicPool.component_part
        ! splicer end class.DynamicPool.component_part
    contains
        procedure :: get_instance => dynamicpool_get_instance
        procedure :: set_instance => dynamicpool_set_instance
        procedure :: associated => dynamicpool_associated
        ! splicer begin class.DynamicPool.type_bound_procedure_part
        ! splicer end class.DynamicPool.type_bound_procedure_part
    end type dynamicpool

    ! splicer begin class.Allocator.module_top
    ! splicer end class.Allocator.module_top

    type, bind(C) :: SHROUD_allocator_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_allocator_capsule

    type UmpireAllocator
        type(SHROUD_allocator_capsule) :: cxxmem
        ! splicer begin class.Allocator.component_part
        ! splicer end class.Allocator.component_part
    contains
        procedure :: allocate => allocator_allocate
        procedure :: deallocate => allocator_deallocate
        procedure :: get_size => allocator_get_size
        procedure :: get_high_watermark => allocator_get_high_watermark
        procedure :: get_current_size => allocator_get_current_size
        procedure :: get_name => allocator_get_name
        procedure :: get_id => allocator_get_id
        procedure :: get_instance => allocator_get_instance
        procedure :: set_instance => allocator_set_instance
        procedure :: associated => allocator_associated
        ! splicer begin class.Allocator.type_bound_procedure_part
        ! splicer end class.Allocator.type_bound_procedure_part
    end type UmpireAllocator

    ! splicer begin class.ResourceManager.module_top
    ! splicer end class.ResourceManager.module_top

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
        procedure :: get_allocatorfor_ptr => resourcemanager_get_allocatorfor_ptr
        procedure :: copy_all => resourcemanager_copy_all
        procedure :: copy_with_size => resourcemanager_copy_with_size
        procedure :: memset_0 => resourcemanager_memset_0
        procedure :: memset_1 => resourcemanager_memset_1
        procedure :: reallocate => resourcemanager_reallocate
        procedure :: deallocate => resourcemanager_deallocate
        procedure :: get_size => resourcemanager_get_size
        procedure :: associated => resourcemanager_associated
        generic :: copy => copy_all, copy_with_size
        generic :: get_allocator => get_allocator_by_name,  &
            get_allocator_by_id, get_allocatorfor_ptr
        generic :: memset => memset_0, memset_1
        ! splicer begin class.ResourceManager.type_bound_procedure_part
        ! splicer end class.ResourceManager.type_bound_procedure_part
    end type UmpireResourceManager

    interface operator (.eq.)
        module procedure dynamicpool_eq
        module procedure allocator_eq
        module procedure resourcemanager_eq
    end interface

    interface operator (.ne.)
        module procedure dynamicpool_ne
        module procedure allocator_ne
        module procedure resourcemanager_ne
    end interface

    interface

        ! splicer begin class.DynamicPool.additional_interfaces
        ! splicer end class.DynamicPool.additional_interfaces

        function c_allocator_allocate(self, bytes) &
                result(SHT_rv) &
                bind(C, name="um_allocator_allocate")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            import :: SHROUD_allocator_capsule
            implicit none
            type(SHROUD_allocator_capsule), intent(IN) :: self
            integer(C_SIZE_T), value, intent(IN) :: bytes
            type(C_PTR) :: SHT_rv
        end function c_allocator_allocate

        subroutine c_allocator_deallocate(self, ptr) &
                bind(C, name="um_allocator_deallocate")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_allocator_capsule
            implicit none
            type(SHROUD_allocator_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
        end subroutine c_allocator_deallocate

        function c_allocator_get_size(self, ptr) &
                result(SHT_rv) &
                bind(C, name="um_allocator_get_size")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            import :: SHROUD_allocator_capsule
            implicit none
            type(SHROUD_allocator_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
            integer(C_SIZE_T) :: SHT_rv
        end function c_allocator_get_size

        function c_allocator_get_high_watermark(self) &
                result(SHT_rv) &
                bind(C, name="um_allocator_get_high_watermark")
            use iso_c_binding, only : C_SIZE_T
            import :: SHROUD_allocator_capsule
            implicit none
            type(SHROUD_allocator_capsule), intent(IN) :: self
            integer(C_SIZE_T) :: SHT_rv
        end function c_allocator_get_high_watermark

        function c_allocator_get_current_size(self) &
                result(SHT_rv) &
                bind(C, name="um_allocator_get_current_size")
            use iso_c_binding, only : C_SIZE_T
            import :: SHROUD_allocator_capsule
            implicit none
            type(SHROUD_allocator_capsule), intent(IN) :: self
            integer(C_SIZE_T) :: SHT_rv
        end function c_allocator_get_current_size

        subroutine c_allocator_get_name_bufferify(self, DSHF_rv) &
                bind(C, name="um_allocator_get_name_bufferify")
            import :: SHROUD_allocator_capsule, SHROUD_array
            implicit none
            type(SHROUD_allocator_capsule), intent(IN) :: self
            type(SHROUD_array), intent(INOUT) :: DSHF_rv
        end subroutine c_allocator_get_name_bufferify

        function c_allocator_get_id(self) &
                result(SHT_rv) &
                bind(C, name="um_allocator_get_id")
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
                bind(C, name="um_resourcemanager_get_instance")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_resourcemanager_get_instance

        function c_resourcemanager_get_allocator_by_name(self, name, &
                SHT_crv) &
                result(SHT_rv) &
                bind(C, name="um_resourcemanager_get_allocator_by_name")
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
                bind(C, name="um_resourcemanager_get_allocator_by_name_bufferify")
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
                bind(C, name="um_resourcemanager_get_allocator_by_id")
            use iso_c_binding, only : C_INT, C_PTR
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: id
            type(SHROUD_allocator_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_resourcemanager_get_allocator_by_id

        function c_resourcemanager_get_allocatorfor_ptr(self, ptr, &
                SHT_crv) &
                result(SHT_rv) &
                bind(C, name="um_resourcemanager_get_allocatorfor_ptr")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_allocator_capsule, SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
            type(SHROUD_allocator_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_resourcemanager_get_allocatorfor_ptr

        subroutine c_resourcemanager_copy_all(self, src_ptr, dst_ptr) &
                bind(C, name="um_resourcemanager_copy_all")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: src_ptr
            type(C_PTR), value, intent(IN) :: dst_ptr
        end subroutine c_resourcemanager_copy_all

        subroutine c_resourcemanager_copy_with_size(self, src_ptr, &
                dst_ptr, size) &
                bind(C, name="um_resourcemanager_copy_with_size")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            import :: SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: src_ptr
            type(C_PTR), value, intent(IN) :: dst_ptr
            integer(C_SIZE_T), value, intent(IN) :: size
        end subroutine c_resourcemanager_copy_with_size

        subroutine c_resourcemanager_memset_0(self, ptr, val) &
                bind(C, name="um_resourcemanager_memset_0")
            use iso_c_binding, only : C_INT, C_PTR
            import :: SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
            integer(C_INT), value, intent(IN) :: val
        end subroutine c_resourcemanager_memset_0

        subroutine c_resourcemanager_memset_1(self, ptr, val, length) &
                bind(C, name="um_resourcemanager_memset_1")
            use iso_c_binding, only : C_INT, C_PTR, C_SIZE_T
            import :: SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
            integer(C_INT), value, intent(IN) :: val
            integer(C_SIZE_T), value, intent(IN) :: length
        end subroutine c_resourcemanager_memset_1

        function c_resourcemanager_reallocate(self, src_ptr, size) &
                result(SHT_rv) &
                bind(C, name="um_resourcemanager_reallocate")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            import :: SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: src_ptr
            integer(C_SIZE_T), value, intent(IN) :: size
            type(C_PTR) :: SHT_rv
        end function c_resourcemanager_reallocate

        subroutine c_resourcemanager_deallocate(self, ptr) &
                bind(C, name="um_resourcemanager_deallocate")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
        end subroutine c_resourcemanager_deallocate

        function c_resourcemanager_get_size(self, ptr) &
                result(SHT_rv) &
                bind(C, name="um_resourcemanager_get_size")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            import :: SHROUD_resourcemanager_capsule
            implicit none
            type(SHROUD_resourcemanager_capsule), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
            integer(C_SIZE_T) :: SHT_rv
        end function c_resourcemanager_get_size

        ! splicer begin class.ResourceManager.additional_interfaces
        ! splicer end class.ResourceManager.additional_interfaces
    end interface

    interface
        ! helper function
        ! Copy the char* or std::string in context into c_var.
        subroutine SHROUD_copy_string_and_free(context, c_var, c_var_size) &
             bind(c,name="um_ShroudCopyStringAndFree")
            use, intrinsic :: iso_c_binding, only : C_CHAR, C_SIZE_T
            import SHROUD_array
            type(SHROUD_array), intent(IN) :: context
            character(kind=C_CHAR), intent(OUT) :: c_var(*)
            integer(C_SIZE_T), value :: c_var_size
        end subroutine SHROUD_copy_string_and_free
    end interface

contains

    ! Return pointer to C++ memory.
    function dynamicpool_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(dynamicpool), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function dynamicpool_get_instance

    subroutine dynamicpool_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(dynamicpool), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine dynamicpool_set_instance

    function dynamicpool_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(dynamicpool), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function dynamicpool_associated

    ! splicer begin class.DynamicPool.additional_functions
    ! splicer end class.DynamicPool.additional_functions

    function allocator_allocate(obj, bytes) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR, C_SIZE_T
        class(UmpireAllocator) :: obj
        integer(C_SIZE_T), value, intent(IN) :: bytes
        type(C_PTR) :: SHT_rv
        ! splicer begin class.Allocator.method.allocate
        SHT_rv = c_allocator_allocate(obj%cxxmem, bytes)
        ! splicer end class.Allocator.method.allocate
    end function allocator_allocate

    subroutine allocator_deallocate(obj, ptr)
        use iso_c_binding, only : C_PTR
        class(UmpireAllocator) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        ! splicer begin class.Allocator.method.deallocate
        call c_allocator_deallocate(obj%cxxmem, ptr)
        ! splicer end class.Allocator.method.deallocate
    end subroutine allocator_deallocate

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

    function resourcemanager_get_allocatorfor_ptr(obj, ptr) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        type(C_PTR) :: SHT_prv
        type(UmpireAllocator) :: SHT_rv
        ! splicer begin class.ResourceManager.method.get_allocatorfor_ptr
        SHT_prv = c_resourcemanager_get_allocatorfor_ptr(obj%cxxmem, &
            ptr, SHT_rv%cxxmem)
        ! splicer end class.ResourceManager.method.get_allocatorfor_ptr
    end function resourcemanager_get_allocatorfor_ptr

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

    subroutine resourcemanager_memset_0(obj, ptr, val)
        use iso_c_binding, only : C_INT, C_PTR
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        integer(C_INT), value, intent(IN) :: val
        ! splicer begin class.ResourceManager.method.memset_0
        call c_resourcemanager_memset_0(obj%cxxmem, ptr, val)
        ! splicer end class.ResourceManager.method.memset_0
    end subroutine resourcemanager_memset_0

    subroutine resourcemanager_memset_1(obj, ptr, val, length)
        use iso_c_binding, only : C_INT, C_PTR, C_SIZE_T
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        integer(C_INT), value, intent(IN) :: val
        integer(C_SIZE_T), value, intent(IN) :: length
        ! splicer begin class.ResourceManager.method.memset_1
        call c_resourcemanager_memset_1(obj%cxxmem, ptr, val, length)
        ! splicer end class.ResourceManager.method.memset_1
    end subroutine resourcemanager_memset_1

    function resourcemanager_reallocate(obj, src_ptr, size) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR, C_SIZE_T
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: src_ptr
        integer(C_SIZE_T), value, intent(IN) :: size
        type(C_PTR) :: SHT_rv
        ! splicer begin class.ResourceManager.method.reallocate
        SHT_rv = c_resourcemanager_reallocate(obj%cxxmem, src_ptr, size)
        ! splicer end class.ResourceManager.method.reallocate
    end function resourcemanager_reallocate

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

    function dynamicpool_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(dynamicpool), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function dynamicpool_eq

    function dynamicpool_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(dynamicpool), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function dynamicpool_ne

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

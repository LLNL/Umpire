! wrapfumpire.f
! This is generated code, do not edit
!>
!! \file wrapfumpire.f
!! \brief Shroud generated wrapper for Umpire library
!<
! splicer begin file_top
! splicer end file_top
module umpire_mod
    use, intrinsic :: iso_c_binding, only : C_PTR
    ! splicer begin module_use
    ! splicer end module_use
    ! splicer begin class.ResourceManager.module_use
    ! splicer end class.ResourceManager.module_use
    ! splicer begin class.Allocator.module_use
    ! splicer end class.Allocator.module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    ! splicer begin class.ResourceManager.module_top
    ! splicer end class.ResourceManager.module_top

    type UmpireResourceManager
        type(C_PTR), private :: voidptr
        ! splicer begin class.ResourceManager.component_part
        ! splicer end class.ResourceManager.component_part
    contains
        procedure :: get_allocator => resourcemanager_get_allocator
        procedure :: copy => resourcemanager_copy
        procedure :: deallocate => resourcemanager_deallocate
        procedure :: get_instance => resourcemanager_get_instance
        procedure :: set_instance => resourcemanager_set_instance
        procedure :: associated => resourcemanager_associated
        ! splicer begin class.ResourceManager.type_bound_procedure_part
        ! splicer end class.ResourceManager.type_bound_procedure_part
    end type UmpireResourceManager

    ! splicer begin class.Allocator.module_top
    ! splicer end class.Allocator.module_top

    type UmpireAllocator
        type(C_PTR), private :: voidptr
        ! splicer begin class.Allocator.component_part
        ! splicer end class.Allocator.component_part
    contains
        procedure :: allocate => allocator_allocate
        procedure :: deallocate => allocator_deallocate
        procedure :: get_instance => allocator_get_instance
        procedure :: set_instance => allocator_set_instance
        procedure :: associated => allocator_associated
        ! splicer begin class.Allocator.type_bound_procedure_part
        ! splicer end class.Allocator.type_bound_procedure_part
    end type UmpireAllocator


    interface operator (.eq.)
        module procedure resourcemanager_eq
        module procedure allocator_eq
    end interface

    interface operator (.ne.)
        module procedure resourcemanager_ne
        module procedure allocator_ne
    end interface

    interface

        function c_resourcemanager_get() &
                result(SH_rv) &
                bind(C, name="UMPIRE_resourcemanager_get")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR) :: SH_rv
        end function c_resourcemanager_get

        function c_resourcemanager_get_allocator(self, space) &
                result(SH_rv) &
                bind(C, name="UMPIRE_resourcemanager_get_allocator")
            use iso_c_binding, only : C_CHAR, C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: space(*)
            type(C_PTR) :: SH_rv
        end function c_resourcemanager_get_allocator

        function c_resourcemanager_get_allocator_bufferify(self, space, Lspace) &
                result(SH_rv) &
                bind(C, name="UMPIRE_resourcemanager_get_allocator_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: space(*)
            integer(C_INT), value, intent(IN) :: Lspace
            type(C_PTR) :: SH_rv
        end function c_resourcemanager_get_allocator_bufferify

        subroutine c_resourcemanager_copy(self, src_ptr, dst_ptr) &
                bind(C, name="UMPIRE_resourcemanager_copy")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            type(C_PTR), value, intent(IN) :: src_ptr
            type(C_PTR), value, intent(IN) :: dst_ptr
        end subroutine c_resourcemanager_copy

        subroutine c_resourcemanager_deallocate(self, ptr) &
                bind(C, name="UMPIRE_resourcemanager_deallocate")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
        end subroutine c_resourcemanager_deallocate

        ! splicer begin class.ResourceManager.additional_interfaces
        ! splicer end class.ResourceManager.additional_interfaces

        function c_allocator_allocate(self, bytes) &
                result(SH_rv) &
                bind(C, name="UMPIRE_allocator_allocate")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            implicit none
            type(C_PTR), value, intent(IN) :: self
            integer(C_SIZE_T), value, intent(IN) :: bytes
            type(C_PTR) :: SH_rv
        end function c_allocator_allocate

        subroutine c_allocator_deallocate(self, ptr) &
                bind(C, name="UMPIRE_allocator_deallocate")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
        end subroutine c_allocator_deallocate

        ! splicer begin class.Allocator.additional_interfaces
        ! splicer end class.Allocator.additional_interfaces
    end interface

contains

    function resourcemanager_get() result(SH_rv)
        type(UmpireResourceManager) :: SH_rv
        ! splicer begin class.ResourceManager.method.get
        SH_rv%voidptr = c_resourcemanager_get()
        ! splicer end class.ResourceManager.method.get
    end function resourcemanager_get

    function resourcemanager_get_allocator(obj, space) result(SH_rv)
        use iso_c_binding, only : C_INT
        class(UmpireResourceManager) :: obj
        character(*), intent(IN) :: space
        type(UmpireAllocator) :: SH_rv
        ! splicer begin class.ResourceManager.method.get_allocator
        SH_rv%voidptr = c_resourcemanager_get_allocator_bufferify(  &
            obj%voidptr,  &
            space,  &
            len_trim(space, kind=C_INT))
        ! splicer end class.ResourceManager.method.get_allocator
    end function resourcemanager_get_allocator

    subroutine resourcemanager_copy(obj, src_ptr, dst_ptr)
        use iso_c_binding, only : C_PTR
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: src_ptr
        type(C_PTR), value, intent(IN) :: dst_ptr
        ! splicer begin class.ResourceManager.method.copy
        call c_resourcemanager_copy(  &
            obj%voidptr,  &
            src_ptr,  &
            dst_ptr)
        ! splicer end class.ResourceManager.method.copy
    end subroutine resourcemanager_copy

    subroutine resourcemanager_deallocate(obj, ptr)
        use iso_c_binding, only : C_PTR
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        ! splicer begin class.ResourceManager.method.deallocate
        call c_resourcemanager_deallocate(  &
            obj%voidptr,  &
            ptr)
        ! splicer end class.ResourceManager.method.deallocate
    end subroutine resourcemanager_deallocate

    function resourcemanager_get_instance(obj) result (voidptr)
        use iso_c_binding, only: C_PTR
        implicit none
        class(UmpireResourceManager), intent(IN) :: obj
        type(C_PTR) :: voidptr
        voidptr = obj%voidptr
    end function resourcemanager_get_instance

    subroutine resourcemanager_set_instance(obj, voidptr)
        use iso_c_binding, only: C_PTR
        implicit none
        class(UmpireResourceManager), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: voidptr
        obj%voidptr = voidptr
    end subroutine resourcemanager_set_instance

    function resourcemanager_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        implicit none
        class(UmpireResourceManager), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%voidptr)
    end function resourcemanager_associated

    ! splicer begin class.ResourceManager.additional_functions
    ! splicer end class.ResourceManager.additional_functions

    function allocator_allocate(obj, bytes) result(SH_rv)
        use iso_c_binding, only : C_PTR, C_SIZE_T
        class(UmpireAllocator) :: obj
        integer(C_SIZE_T), value, intent(IN) :: bytes
        type(C_PTR) :: SH_rv
        ! splicer begin class.Allocator.method.allocate
        SH_rv = c_allocator_allocate(  &
            obj%voidptr,  &
            bytes)
        ! splicer end class.Allocator.method.allocate
    end function allocator_allocate

    subroutine allocator_deallocate(obj, ptr)
        use iso_c_binding, only : C_PTR
        class(UmpireAllocator) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        ! splicer begin class.Allocator.method.deallocate
        call c_allocator_deallocate(  &
            obj%voidptr,  &
            ptr)
        ! splicer end class.Allocator.method.deallocate
    end subroutine allocator_deallocate

    function allocator_get_instance(obj) result (voidptr)
        use iso_c_binding, only: C_PTR
        implicit none
        class(UmpireAllocator), intent(IN) :: obj
        type(C_PTR) :: voidptr
        voidptr = obj%voidptr
    end function allocator_get_instance

    subroutine allocator_set_instance(obj, voidptr)
        use iso_c_binding, only: C_PTR
        implicit none
        class(UmpireAllocator), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: voidptr
        obj%voidptr = voidptr
    end subroutine allocator_set_instance

    function allocator_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        implicit none
        class(UmpireAllocator), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%voidptr)
    end function allocator_associated

    ! splicer begin class.Allocator.additional_functions
    ! splicer end class.Allocator.additional_functions

    function resourcemanager_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        implicit none
        type(UmpireResourceManager), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%voidptr, b%voidptr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function resourcemanager_eq

    function resourcemanager_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        implicit none
        type(UmpireResourceManager), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%voidptr, b%voidptr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function resourcemanager_ne

    function allocator_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        implicit none
        type(UmpireAllocator), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%voidptr, b%voidptr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function allocator_eq

    function allocator_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        implicit none
        type(UmpireAllocator), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%voidptr, b%voidptr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function allocator_ne

end module umpire_mod

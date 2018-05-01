! wrapfumpire.f
! This is generated code, do not edit
!>
!! \file wrapfumpire.f
!! \brief Shroud generated wrapper for Umpire library
!<
! splicer begin file_top
! splicer end file_top
module umpire_mod
    use iso_c_binding, only : C_PTR
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    ! splicer begin class.Allocator.module_top
    ! splicer end class.Allocator.module_top

    type UmpireAllocator
        type(C_PTR), private :: voidptr
        ! splicer begin class.Allocator.component_part
        ! splicer end class.Allocator.component_part
    contains
        procedure :: allocate => allocator_allocate
        procedure :: deallocate => allocator_deallocate
        procedure :: get_size => allocator_get_size
        procedure :: get_high_watermark => allocator_get_high_watermark
        procedure :: get_current_size => allocator_get_current_size
        procedure :: get_id => allocator_get_id
        procedure :: get_instance => allocator_get_instance
        procedure :: set_instance => allocator_set_instance
        procedure :: associated => allocator_associated
        ! splicer begin class.Allocator.type_bound_procedure_part
        ! splicer end class.Allocator.type_bound_procedure_part
    end type UmpireAllocator

    ! splicer begin class.ResourceManager.module_top
    ! splicer end class.ResourceManager.module_top

    type UmpireResourceManager
        type(C_PTR), private :: voidptr
        ! splicer begin class.ResourceManager.component_part
        ! splicer end class.ResourceManager.component_part
    contains
        procedure, nopass :: getinstance => resourcemanager_getinstance
        procedure :: initialize => resourcemanager_initialize
        procedure :: get_allocator_0 => resourcemanager_get_allocator_0
        procedure :: get_allocator_1 => resourcemanager_get_allocator_1
        procedure, nopass :: delete_allocator => resourcemanager_delete_allocator
        procedure :: copy_0 => resourcemanager_copy_0
        procedure :: copy_1 => resourcemanager_copy_1
        procedure :: memset_0 => resourcemanager_memset_0
        procedure :: memset_1 => resourcemanager_memset_1
        procedure :: reallocate => resourcemanager_reallocate
        procedure :: deallocate => resourcemanager_deallocate
        procedure :: get_size => resourcemanager_get_size
        procedure :: get_instance => resourcemanager_get_instance
        procedure :: set_instance => resourcemanager_set_instance
        procedure :: associated => resourcemanager_associated
        generic :: copy => copy_0, copy_1
        generic :: get_allocator => get_allocator_0, get_allocator_1
        generic :: memset => memset_0, memset_1
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

        function c_allocator_allocate(self, bytes) &
                result(SHT_rv) &
                bind(C, name="umpire_allocator_allocate")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            implicit none
            type(C_PTR), value, intent(IN) :: self
            integer(C_SIZE_T), value, intent(IN) :: bytes
            type(C_PTR) :: SHT_rv
        end function c_allocator_allocate

        subroutine c_allocator_deallocate(self, ptr) &
                bind(C, name="umpire_allocator_deallocate")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
        end subroutine c_allocator_deallocate

        function c_allocator_get_size(self, ptr) &
                result(SHT_rv) &
                bind(C, name="umpire_allocator_get_size")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            implicit none
            type(C_PTR), value, intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
            integer(C_SIZE_T) :: SHT_rv
        end function c_allocator_get_size

        function c_allocator_get_high_watermark(self) &
                result(SHT_rv) &
                bind(C, name="umpire_allocator_get_high_watermark")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            implicit none
            type(C_PTR), value, intent(IN) :: self
            integer(C_SIZE_T) :: SHT_rv
        end function c_allocator_get_high_watermark

        function c_allocator_get_current_size(self) &
                result(SHT_rv) &
                bind(C, name="umpire_allocator_get_current_size")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            implicit none
            type(C_PTR), value, intent(IN) :: self
            integer(C_SIZE_T) :: SHT_rv
        end function c_allocator_get_current_size

        function c_allocator_get_id(self) &
                result(SHT_rv) &
                bind(C, name="umpire_allocator_get_id")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            implicit none
            type(C_PTR), value, intent(IN) :: self
            integer(C_SIZE_T) :: SHT_rv
        end function c_allocator_get_id

        ! splicer begin class.Allocator.additional_interfaces
        ! splicer end class.Allocator.additional_interfaces

        function c_resourcemanager_getinstance() &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_getinstance")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR) :: SHT_rv
        end function c_resourcemanager_getinstance

        subroutine c_resourcemanager_initialize(self) &
                bind(C, name="umpire_resourcemanager_initialize")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
        end subroutine c_resourcemanager_initialize

        function c_resourcemanager_get_allocator_0(self, name) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_get_allocator_0")
            use iso_c_binding, only : C_CHAR, C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            type(C_PTR) :: SHT_rv
        end function c_resourcemanager_get_allocator_0

        function c_resourcemanager_get_allocator_0_bufferify(self, name, &
                Lname) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_get_allocator_0_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
            type(C_PTR) :: SHT_rv
        end function c_resourcemanager_get_allocator_0_bufferify

        function c_resourcemanager_get_allocator_1(self, id) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_get_allocator_1")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            integer(C_INT), value, intent(IN) :: id
            type(C_PTR) :: SHT_rv
        end function c_resourcemanager_get_allocator_1

        subroutine c_resourcemanager_delete_allocator(alloc_obj) &
                bind(C, name="umpire_resourcemanager_delete_allocator")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: alloc_obj
        end subroutine c_resourcemanager_delete_allocator

        subroutine c_resourcemanager_copy_0(self, src_ptr, dst_ptr) &
                bind(C, name="umpire_resourcemanager_copy_0")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            type(C_PTR), value, intent(IN) :: src_ptr
            type(C_PTR), value, intent(IN) :: dst_ptr
        end subroutine c_resourcemanager_copy_0

        subroutine c_resourcemanager_copy_1(self, src_ptr, dst_ptr, &
                size) &
                bind(C, name="umpire_resourcemanager_copy_1")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            implicit none
            type(C_PTR), value, intent(IN) :: self
            type(C_PTR), value, intent(IN) :: src_ptr
            type(C_PTR), value, intent(IN) :: dst_ptr
            integer(C_SIZE_T), value, intent(IN) :: size
        end subroutine c_resourcemanager_copy_1

        subroutine c_resourcemanager_memset_0(self, ptr, val) &
                bind(C, name="umpire_resourcemanager_memset_0")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
            integer(C_INT), value, intent(IN) :: val
        end subroutine c_resourcemanager_memset_0

        subroutine c_resourcemanager_memset_1(self, ptr, val, length) &
                bind(C, name="umpire_resourcemanager_memset_1")
            use iso_c_binding, only : C_INT, C_PTR, C_SIZE_T
            implicit none
            type(C_PTR), value, intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
            integer(C_INT), value, intent(IN) :: val
            integer(C_SIZE_T), value, intent(IN) :: length
        end subroutine c_resourcemanager_memset_1

        function c_resourcemanager_reallocate(self, src_ptr, size) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_reallocate")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            implicit none
            type(C_PTR), value, intent(IN) :: self
            type(C_PTR), value, intent(IN) :: src_ptr
            integer(C_SIZE_T), value, intent(IN) :: size
            type(C_PTR) :: SHT_rv
        end function c_resourcemanager_reallocate

        subroutine c_resourcemanager_deallocate(self, ptr) &
                bind(C, name="umpire_resourcemanager_deallocate")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
        end subroutine c_resourcemanager_deallocate

        function c_resourcemanager_get_size(self, ptr) &
                result(SHT_rv) &
                bind(C, name="umpire_resourcemanager_get_size")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            implicit none
            type(C_PTR), value, intent(IN) :: self
            type(C_PTR), value, intent(IN) :: ptr
            integer(C_SIZE_T) :: SHT_rv
        end function c_resourcemanager_get_size

        ! splicer begin class.ResourceManager.additional_interfaces
        ! splicer end class.ResourceManager.additional_interfaces
    end interface

contains

    function allocator_allocate(obj, bytes) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR, C_SIZE_T
        class(UmpireAllocator) :: obj
        integer(C_SIZE_T), value, intent(IN) :: bytes
        type(C_PTR) :: SHT_rv
        ! splicer begin class.Allocator.method.allocate
        SHT_rv = c_allocator_allocate(obj%voidptr, bytes)
        ! splicer end class.Allocator.method.allocate
    end function allocator_allocate

    subroutine allocator_deallocate(obj, ptr)
        use iso_c_binding, only : C_PTR
        class(UmpireAllocator) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        ! splicer begin class.Allocator.method.deallocate
        call c_allocator_deallocate(obj%voidptr, ptr)
        ! splicer end class.Allocator.method.deallocate
    end subroutine allocator_deallocate

    function allocator_get_size(obj, ptr) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR, C_SIZE_T
        class(UmpireAllocator) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        integer(C_SIZE_T) :: SHT_rv
        ! splicer begin class.Allocator.method.get_size
        SHT_rv = c_allocator_get_size(obj%voidptr, ptr)
        ! splicer end class.Allocator.method.get_size
    end function allocator_get_size

    function allocator_get_high_watermark(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_SIZE_T
        class(UmpireAllocator) :: obj
        integer(C_SIZE_T) :: SHT_rv
        ! splicer begin class.Allocator.method.get_high_watermark
        SHT_rv = c_allocator_get_high_watermark(obj%voidptr)
        ! splicer end class.Allocator.method.get_high_watermark
    end function allocator_get_high_watermark

    function allocator_get_current_size(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_SIZE_T
        class(UmpireAllocator) :: obj
        integer(C_SIZE_T) :: SHT_rv
        ! splicer begin class.Allocator.method.get_current_size
        SHT_rv = c_allocator_get_current_size(obj%voidptr)
        ! splicer end class.Allocator.method.get_current_size
    end function allocator_get_current_size

    function allocator_get_id(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_SIZE_T
        class(UmpireAllocator) :: obj
        integer(C_SIZE_T) :: SHT_rv
        ! splicer begin class.Allocator.method.get_id
        SHT_rv = c_allocator_get_id(obj%voidptr)
        ! splicer end class.Allocator.method.get_id
    end function allocator_get_id

    function allocator_get_instance(obj) result (voidptr)
        use iso_c_binding, only: C_PTR
        class(UmpireAllocator), intent(IN) :: obj
        type(C_PTR) :: voidptr
        voidptr = obj%voidptr
    end function allocator_get_instance

    subroutine allocator_set_instance(obj, voidptr)
        use iso_c_binding, only: C_PTR
        class(UmpireAllocator), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: voidptr
        obj%voidptr = voidptr
    end subroutine allocator_set_instance

    function allocator_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(UmpireAllocator), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%voidptr)
    end function allocator_associated

    ! splicer begin class.Allocator.additional_functions
    ! splicer end class.Allocator.additional_functions

    function resourcemanager_getinstance() &
            result(SHT_rv)
        type(UmpireResourceManager) :: SHT_rv
        ! splicer begin class.ResourceManager.method.getinstance
        SHT_rv%voidptr = c_resourcemanager_getinstance()
        ! splicer end class.ResourceManager.method.getinstance
    end function resourcemanager_getinstance

    subroutine resourcemanager_initialize(obj)
        class(UmpireResourceManager) :: obj
        ! splicer begin class.ResourceManager.method.initialize
        call c_resourcemanager_initialize(obj%voidptr)
        ! splicer end class.ResourceManager.method.initialize
    end subroutine resourcemanager_initialize

    function resourcemanager_get_allocator_0(obj, name) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(UmpireResourceManager) :: obj
        character(*), intent(IN) :: name
        type(UmpireAllocator) :: SHT_rv
        ! splicer begin class.ResourceManager.method.get_allocator_0
        SHT_rv%voidptr = c_resourcemanager_get_allocator_0_bufferify(obj%voidptr, &
            name, len_trim(name, kind=C_INT))
        ! splicer end class.ResourceManager.method.get_allocator_0
    end function resourcemanager_get_allocator_0

    function resourcemanager_get_allocator_1(obj, id) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(UmpireResourceManager) :: obj
        integer(C_INT), value, intent(IN) :: id
        type(UmpireAllocator) :: SHT_rv
        ! splicer begin class.ResourceManager.method.get_allocator_1
        SHT_rv%voidptr = c_resourcemanager_get_allocator_1(obj%voidptr, &
            id)
        ! splicer end class.ResourceManager.method.get_allocator_1
    end function resourcemanager_get_allocator_1

    subroutine resourcemanager_delete_allocator(alloc_obj)
        type(UmpireAllocator), value, intent(IN) :: alloc_obj
        ! splicer begin class.ResourceManager.method.delete_allocator
        call c_resourcemanager_delete_allocator(alloc_obj%get_instance())
        ! splicer end class.ResourceManager.method.delete_allocator
    end subroutine resourcemanager_delete_allocator

    subroutine resourcemanager_copy_0(obj, src_ptr, dst_ptr)
        use iso_c_binding, only : C_PTR
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: src_ptr
        type(C_PTR), value, intent(IN) :: dst_ptr
        ! splicer begin class.ResourceManager.method.copy_0
        call c_resourcemanager_copy_0(obj%voidptr, src_ptr, dst_ptr)
        ! splicer end class.ResourceManager.method.copy_0
    end subroutine resourcemanager_copy_0

    subroutine resourcemanager_copy_1(obj, src_ptr, dst_ptr, size)
        use iso_c_binding, only : C_PTR, C_SIZE_T
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: src_ptr
        type(C_PTR), value, intent(IN) :: dst_ptr
        integer(C_SIZE_T), value, intent(IN) :: size
        ! splicer begin class.ResourceManager.method.copy_1
        call c_resourcemanager_copy_1(obj%voidptr, src_ptr, dst_ptr, &
            size)
        ! splicer end class.ResourceManager.method.copy_1
    end subroutine resourcemanager_copy_1

    subroutine resourcemanager_memset_0(obj, ptr, val)
        use iso_c_binding, only : C_INT, C_PTR
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        integer(C_INT), value, intent(IN) :: val
        ! splicer begin class.ResourceManager.method.memset_0
        call c_resourcemanager_memset_0(obj%voidptr, ptr, val)
        ! splicer end class.ResourceManager.method.memset_0
    end subroutine resourcemanager_memset_0

    subroutine resourcemanager_memset_1(obj, ptr, val, length)
        use iso_c_binding, only : C_INT, C_PTR, C_SIZE_T
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        integer(C_INT), value, intent(IN) :: val
        integer(C_SIZE_T), value, intent(IN) :: length
        ! splicer begin class.ResourceManager.method.memset_1
        call c_resourcemanager_memset_1(obj%voidptr, ptr, val, length)
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
        SHT_rv = c_resourcemanager_reallocate(obj%voidptr, src_ptr, &
            size)
        ! splicer end class.ResourceManager.method.reallocate
    end function resourcemanager_reallocate

    subroutine resourcemanager_deallocate(obj, ptr)
        use iso_c_binding, only : C_PTR
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        ! splicer begin class.ResourceManager.method.deallocate
        call c_resourcemanager_deallocate(obj%voidptr, ptr)
        ! splicer end class.ResourceManager.method.deallocate
    end subroutine resourcemanager_deallocate

    function resourcemanager_get_size(obj, ptr) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR, C_SIZE_T
        class(UmpireResourceManager) :: obj
        type(C_PTR), value, intent(IN) :: ptr
        integer(C_SIZE_T) :: SHT_rv
        ! splicer begin class.ResourceManager.method.get_size
        SHT_rv = c_resourcemanager_get_size(obj%voidptr, ptr)
        ! splicer end class.ResourceManager.method.get_size
    end function resourcemanager_get_size

    function resourcemanager_get_instance(obj) result (voidptr)
        use iso_c_binding, only: C_PTR
        class(UmpireResourceManager), intent(IN) :: obj
        type(C_PTR) :: voidptr
        voidptr = obj%voidptr
    end function resourcemanager_get_instance

    subroutine resourcemanager_set_instance(obj, voidptr)
        use iso_c_binding, only: C_PTR
        class(UmpireResourceManager), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: voidptr
        obj%voidptr = voidptr
    end subroutine resourcemanager_set_instance

    function resourcemanager_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(UmpireResourceManager), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%voidptr)
    end function resourcemanager_associated

    ! splicer begin class.ResourceManager.additional_functions
    ! splicer end class.ResourceManager.additional_functions

    function allocator_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
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
        type(UmpireAllocator), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%voidptr, b%voidptr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function allocator_ne

    function resourcemanager_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
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
        type(UmpireResourceManager), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%voidptr, b%voidptr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function resourcemanager_ne

end module umpire_mod

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
!! project contributors. See the COPYRIGHT file for details.
!!
!! SPDX-License-Identifier: (MIT)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! wrapfUmpire_strategy.f
! This is generated code, do not edit
! Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
! project contributors. See the COPYRIGHT file for details.
!
! SPDX-License-Identifier: (MIT)
!>
!! \file wrapfUmpire_strategy.f
!! \brief Shroud generated wrapper for strategy namespace
!<
! splicer begin namespace.strategy.file_top
! splicer end namespace.strategy.file_top
module umpire_strategy_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin namespace.strategy.module_use
    ! splicer end namespace.strategy.module_use
    implicit none

    ! splicer begin namespace.strategy.module_top
    ! splicer end namespace.strategy.module_top

    type, bind(C) :: SHROUD_dynamicpool_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_dynamicpool_capsule

    type dynamicpool
        type(SHROUD_dynamicpool_capsule) :: cxxmem
        ! splicer begin namespace.strategy.class.DynamicPool.component_part
        ! splicer end namespace.strategy.class.DynamicPool.component_part
    contains
        procedure :: get_instance => dynamicpool_get_instance
        procedure :: set_instance => dynamicpool_set_instance
        procedure :: associated => dynamicpool_associated
        ! splicer begin namespace.strategy.class.DynamicPool.type_bound_procedure_part
        ! splicer end namespace.strategy.class.DynamicPool.type_bound_procedure_part
    end type dynamicpool

    type, bind(C) :: SHROUD_allocationadvisor_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_allocationadvisor_capsule

    type allocationadvisor
        type(SHROUD_allocationadvisor_capsule) :: cxxmem
        ! splicer begin namespace.strategy.class.AllocationAdvisor.component_part
        ! splicer end namespace.strategy.class.AllocationAdvisor.component_part
    contains
        procedure :: get_instance => allocationadvisor_get_instance
        procedure :: set_instance => allocationadvisor_set_instance
        procedure :: associated => allocationadvisor_associated
        ! splicer begin namespace.strategy.class.AllocationAdvisor.type_bound_procedure_part
        ! splicer end namespace.strategy.class.AllocationAdvisor.type_bound_procedure_part
    end type allocationadvisor

    type, bind(C) :: SHROUD_namedallocationstrategy_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_namedallocationstrategy_capsule

    type namedallocationstrategy
        type(SHROUD_namedallocationstrategy_capsule) :: cxxmem
        ! splicer begin namespace.strategy.class.NamedAllocationStrategy.component_part
        ! splicer end namespace.strategy.class.NamedAllocationStrategy.component_part
    contains
        procedure :: get_instance => namedallocationstrategy_get_instance
        procedure :: set_instance => namedallocationstrategy_set_instance
        procedure :: associated => namedallocationstrategy_associated
        ! splicer begin namespace.strategy.class.NamedAllocationStrategy.type_bound_procedure_part
        ! splicer end namespace.strategy.class.NamedAllocationStrategy.type_bound_procedure_part
    end type namedallocationstrategy

    interface operator (.eq.)
        module procedure dynamicpool_eq
        module procedure allocationadvisor_eq
        module procedure namedallocationstrategy_eq
    end interface

    interface operator (.ne.)
        module procedure dynamicpool_ne
        module procedure allocationadvisor_ne
        module procedure namedallocationstrategy_ne
    end interface

    interface

        ! splicer begin namespace.strategy.class.DynamicPool.additional_interfaces
        ! splicer end namespace.strategy.class.DynamicPool.additional_interfaces

        ! splicer begin namespace.strategy.class.AllocationAdvisor.additional_interfaces
        ! splicer end namespace.strategy.class.AllocationAdvisor.additional_interfaces

        ! splicer begin namespace.strategy.class.NamedAllocationStrategy.additional_interfaces
        ! splicer end namespace.strategy.class.NamedAllocationStrategy.additional_interfaces

        ! splicer begin namespace.strategy.additional_interfaces
        ! splicer end namespace.strategy.additional_interfaces
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

    ! splicer begin namespace.strategy.class.DynamicPool.additional_functions
    ! splicer end namespace.strategy.class.DynamicPool.additional_functions

    ! Return pointer to C++ memory.
    function allocationadvisor_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(allocationadvisor), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function allocationadvisor_get_instance

    subroutine allocationadvisor_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(allocationadvisor), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine allocationadvisor_set_instance

    function allocationadvisor_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(allocationadvisor), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function allocationadvisor_associated

    ! splicer begin namespace.strategy.class.AllocationAdvisor.additional_functions
    ! splicer end namespace.strategy.class.AllocationAdvisor.additional_functions

    ! Return pointer to C++ memory.
    function namedallocationstrategy_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(namedallocationstrategy), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function namedallocationstrategy_get_instance

    subroutine namedallocationstrategy_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(namedallocationstrategy), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine namedallocationstrategy_set_instance

    function namedallocationstrategy_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(namedallocationstrategy), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function namedallocationstrategy_associated

    ! splicer begin namespace.strategy.class.NamedAllocationStrategy.additional_functions
    ! splicer end namespace.strategy.class.NamedAllocationStrategy.additional_functions

    ! splicer begin namespace.strategy.additional_functions
    ! splicer end namespace.strategy.additional_functions

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

    function allocationadvisor_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(allocationadvisor), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function allocationadvisor_eq

    function allocationadvisor_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(allocationadvisor), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function allocationadvisor_ne

    function namedallocationstrategy_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(namedallocationstrategy), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function namedallocationstrategy_eq

    function namedallocationstrategy_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(namedallocationstrategy), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function namedallocationstrategy_ne

end module umpire_strategy_mod

set (allocator_tests_depends
  umpire umpire_alloc gtest)

if (ENABLE_CUDA)
  set (allocator_tests_depends
    ${allocator_tests_depends}
    umpire_tpl_cnmem 
    cuda_runtime)
endif ()

blt_add_executable(
  NAME allocator_tests
  SOURCES allocator_tests.cpp
  DEPENDS_ON ${allocator_tests_depends}
  OUTPUT_DIR ${UMPIRE_TEST_OUTPUT_DIR})

target_include_directories(
  allocator_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME allocator_tests
  COMMAND allocator_tests)

blt_add_executable(
  NAME operation_tests
  SOURCES operation_tests.cpp
  DEPENDS_ON umpire gtest
  OUTPUT_DIR ${UMPIRE_TEST_OUTPUT_DIR})

target_include_directories(
  operation_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME operation_tests
  COMMAND operation_tests)

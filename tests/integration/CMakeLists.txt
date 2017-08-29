blt_add_executable(
  NAME allocator_tests
  SOURCES allocator_tests.cpp
  DEPENDS_ON umpire umpire_tpl_cnmem umpire_alloc gtest cuda_runtime
  OUTPUT_DIR ${UMPIRE_TEST_OUTPUT_DIR})

target_include_directories(
  allocator_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME allocator_tests
  COMMAND allocator_tests)

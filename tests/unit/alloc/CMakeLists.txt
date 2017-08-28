blt_add_executable(
  NAME memory_allocator_tests
  SOURCES memory_allocator_tests.cpp
  DEPENDS_ON umpire_alloc gtest umpire_tpl_cnmem cuda_runtime
  OUTPUT_DIR ${UMPIRE_TEST_OUTPUT_DIR})

target_include_directories(
  memory_allocator_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME memory_allocator_tests
  COMMAND memory_allocator_tests)

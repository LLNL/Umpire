set (memory_allocator_tests_depends
  umpire_alloc gtest)

if (ENABLE_CUDA)
  set (memory_allocator_tests_depends
    ${memory_allocator_tests_depends}
    cuda_runtime)
endif ()

blt_add_executable(
  NAME memory_allocator_tests
  SOURCES memory_allocator_tests.cpp
  DEPENDS_ON ${memory_allocator_tests_depends}
  OUTPUT_DIR ${UMPIRE_TEST_OUTPUT_DIR})

target_include_directories(
  memory_allocator_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME memory_allocator_tests
  COMMAND memory_allocator_tests)

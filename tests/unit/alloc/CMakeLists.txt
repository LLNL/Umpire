blt_add_executable(
  NAME memory_allocator_tests
  SOURCES memory_allocator_tests.cpp
  DEPENDS_ON umpire_alloc gtest
  OUTPUT_DIR ${UMPIRE_TEST_OUTPUT_DIR})

blt_add_test(
  NAME memory_allocator_tests
  COMMAND memory_allocator_tests)

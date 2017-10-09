blt_add_executable(
  NAME memoryspace_tests
  SOURCES memoryspace_tests.cpp
  DEPENDS_ON umpire_space umpire_util umpire gtest gmock
  OUTPUT_DIR ${UMPIRE_TEST_OUTPUT_DIR})

blt_add_test(
  NAME memoryspace_tests
  COMMAND memoryspace_tests)

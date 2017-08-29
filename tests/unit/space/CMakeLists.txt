blt_add_executable(
  NAME space_tests
  SOURCES space_tests.cpp
  DEPENDS_ON umpire_space umpire_util umpire gtest gmock)

blt_add_test(
  NAME space_tests
  COMMAND space_tests)

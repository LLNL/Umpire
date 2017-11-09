blt_add_executable(
  NAME resource_tests
  SOURCES resource_tests.cpp
  DEPENDS_ON umpire_resource umpire_util umpire gtest gmock)

blt_add_test(
  NAME resource_tests
  COMMAND resource_tests)

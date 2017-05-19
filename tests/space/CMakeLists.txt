blt_add_executable(
  NAME space_registry_tests
  SOURCES space_registry_tests.cpp
  DEPENDS_ON umpire umpire_space umpire_util gtest)

blt_add_test(
  NAME space_registry_tests
  COMMAND space_registry_tests)


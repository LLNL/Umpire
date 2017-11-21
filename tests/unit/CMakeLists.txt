blt_add_executable(
  NAME resource_manager_tests
  SOURCES resource_manager_tests.cpp
  DEPENDS_ON umpire gtest)

blt_add_test(
  NAME resource_manager_tests
  COMMAND resource_manager_tests)

blt_add_executable(
  NAME allocator_registry_tests
  SOURCES allocator_registry_tests.cpp
  DEPENDS_ON umpire gtest gmock
  OUTPUT_DIR ${UMPIRE_TEST_OUTPUT_DIR}
)

blt_add_test(
  NAME allocator_registry_tests
  COMMAND allocator_registry_tests)

add_subdirectory(alloc)
add_subdirectory(resource)
add_subdirectory(util)

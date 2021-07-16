#pragma once

#include "umpire/detail/registry.hpp"

#include "umpire/alloc.hpp"
#include "umpire/resource.hpp"
#include "umpire/strategy.hpp"
#include "umpire/allocator.hpp"
#include "umpire/op.hpp"

#include <memory>

namespace umpire {

void initialize();
void finalize();

inline
int get_major_version()
{
  return UMPIRE_VERSION_MAJOR;
}

inline
int get_minor_version()
{
  return UMPIRE_VERSION_MINOR;
}

inline
int get_patch_version()
{
  return UMPIRE_VERSION_PATCH;
}

inline
std::string get_rc_version()
{
  return UMPIRE_VERSION_RC;
}

/*!
 * \brief Print the allocations from a specific allocator in a
 * human-readable format.
 *
 * \param allocator source Allocator.
 * \param os output stream
 */
void print_allocator_records(Allocator allocator, std::ostream& os = std::cout);

/*!
 * \brief Returns vector of AllocationRecords created by the allocator.
 *
 * \param allocator source Allocator.
 */
std::vector<allocation_record> get_allocator_records(Allocator allocator);

/*!
 * \brief Check whether the right allocation overlaps the left.
 * 
 * right will overlap left if the right is greater than left, but less than left+size, and right+size is strictly greater than left+size.
 * 
 * \param left Pointer to left allocation
 * \param right Poniter to right allocation
 */
bool pointer_overlaps(void* left, void* right);

/*!
 * \brief Check whether the left allocation contains the right.
 * 
 * right is contained by left if right is greater than left, and right+size is greater than left+size.
 * 
 * \param left Pointer to left allocation
 * \param right Poniter to right allocation
 */
bool pointer_contains(void* left, void* right);

/*!
 * \brief Get the backtrace associated with the allocation of ptr
 *
 * The string may be empty if backtraces are not enabled.
 */
std::string get_backtrace(void* ptr);

/*!
 * \brief Get memory usage of the current process (uses underlying system-dependent calls)
 */
std::size_t get_process_memory_usage();

/*!
 * \brief Get memory usage of device device_id, using appropriate underlying vendor API.
 */ 
std::size_t get_device_memory_usage(int device_id);

/*!
 * \brief Get all the leaked (active) allocations associated with allocator.
 */
std::vector<allocation_record> get_leaked_allocations(Allocator allocator);

memory* get_strategy(const std::string& name);

template<typename Strategy>
inline Strategy* register_strategy(const std::string& name, std::unique_ptr<Strategy>&& strategy) {
  auto& list = detail::registry::get()->get_allocator_list();
  auto& by_name = detail::registry::get()->get_allocator_name_map();
  auto& by_id = detail::registry::get()->get_allocator_id_map();

  Strategy* s = strategy.get();
  by_name[name] = s;
  by_id[strategy->get_id()] = s;
  list.emplace_front(std::move(strategy));

  return s;
}

template<typename Strategy>
Strategy* get_strategy(const std::string& name)
{
  auto& allocators = detail::registry::get()->get_allocator_name_map();
  auto allocator = allocators.find(name);

  if (allocator == allocators.end()) {
    UMPIRE_ERROR("Allocator \"" << name << "\" not found. Available allocators: ");
        //<< getAllocatorInformation());
  }

  return dynamic_cast<Strategy*>(allocator->second);
}

Allocator get_allocator(const std::string& name);

void register_allocator(const std::string& name, Allocator allocator);

int get_device_count();

bool is_allocator(const std::string& name);

template <typename T>
inline allocator<T> get_allocator(const std::string& name) {
  return allocator<T>(detail::registry::get()->get_allocator_name_map()[name]);
}

template <typename T>
inline allocator<T> get_allocator(int)
{}

template <typename Strategy, typename... Args>
inline Strategy* make_strategy(const std::string& name, Args&&... args)
{
  auto strategy = std::make_unique<Strategy>(name, std::forward<Args>(args)...);
  return register_strategy(name, std::move(strategy));
}

// template <typename Strategy, typename... Args>
// inline Allocator make_allocator(const std::string& name, Args&&... args)
// {
//   return Allocator(make_strategy<Strategy>(name, std::forward<Args...>(args...)()));
// }

template<typename T, typename Strategy>
struct allocator_proxy
{
  constexpr allocator_proxy(Strategy* s) :
    strategy_{s} {}

  constexpr operator allocator<T>() const {
    return allocator<T>(strategy_);
  } 

  constexpr operator allocator<T, Strategy>() const {
    return allocator<T, Strategy>(strategy_);
  }

  private:
    Strategy* strategy_;
};

// template <typename T, typename Strategy, typename... Args>
// inline auto make_allocator_t(const std::string& name, Args&&... args) -> allocator<T, Strategy>
// {
//   return allocator<T, Strategy>(make_strategy<Strategy>(name, std::forward<Args>(args)...));
// }

template <typename T, typename Strategy, typename... Args>
inline auto make_allocator(const std::string& name, Args&&... args) -> allocator_proxy<T, Strategy>
{
  return allocator_proxy<T, Strategy>(make_strategy<Strategy>(name, std::forward<Args>(args)...));
}

template <typename T, typename Strategy=memory>
Allocator get_allocator(T* ptr)
{
  const auto map = detail::registry::get()->get_allocation_map();
  const auto record = map.find(ptr);
  return allocator_proxy<T, Strategy>(record->allocation_strategy);
}

template <typename T>
std::size_t get_size(T* ptr)
{
  const auto map = detail::registry::get()->get_allocation_map();
  const auto record = map.find(ptr);
  return record->size;
}


template <typename T, typename Memory>
void register_allocator(const std::string& name, allocator<T, Memory> allocator)
{
  auto memory = allocator.get_memory();
}

template <typename T>
bool has_allocator(T* ptr) { 
  const auto map = detail::registry::get()->get_allocation_map();
  return (map.findRecord(ptr) != nullptr);
}

std::vector<std::string> get_allocator_names();
std::vector<int> get_allocator_ids();
bool is_allocator(const std::string& name);

}

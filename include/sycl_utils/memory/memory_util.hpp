#ifndef SYCL_UTILS_SYCL_MEMORY_UTIL_HPP_
#define SYCL_UTILS_SYCL_MEMORY_UTIL_HPP_

#include <CL/sycl.hpp>
#include <iostream>

// accessors type aliases
// read accessor
template <typename T, size_t Dimension,
          cl::sycl::access::target Target =
              cl::sycl::access::target::global_buffer>
using read_accessor_t =
    cl::sycl::accessor<T, Dimension, cl::sycl::access::mode::read, Target>;
// write accessor
template <typename T, size_t Dimension,
          cl::sycl::access::target Target =
              cl::sycl::access::target::global_buffer>
using write_accessor_t =
    cl::sycl::accessor<T, Dimension, cl::sycl::access::mode::write, Target>;
// read and write accessor
template <typename T, size_t Dimension,
          cl::sycl::access::target Target =
              cl::sycl::access::target::global_buffer>
using read_write_accessor_t =
    cl::sycl::accessor<T, Dimension, cl::sycl::access::mode::read_write,
                       Target>;
// discard write accessor type for global memory
template <typename T, size_t Dimension,
          cl::sycl::access::target Target =
              cl::sycl::access::target::global_buffer>
using discard_write_accessor_t =
    cl::sycl::accessor<T, Dimension, cl::sycl::access::mode::discard_write,
                       Target>;
// discard read and write accessor type for global memory
template <typename T, size_t Dimension,
          cl::sycl::access::target Target =
              cl::sycl::access::target::global_buffer>
using discard_read_write_accessor_t =
    cl::sycl::accessor<T, Dimension, cl::sycl::access::mode::discard_read_write,
                       Target>;

// ...
static constexpr auto sycl_mode_read = cl::sycl::access::mode::read;
static constexpr auto sycl_mode_write = cl::sycl::access::mode::write;
static constexpr auto sycl_mode_read_write = cl::sycl::access::mode::read_write;

// ...
static constexpr auto sycl_target_local = cl::sycl::access::target::local;

// computes the highest power of 2 number of compute units
static size_t roundup_cu(const cl::sycl::queue &queue) {
  /* TODO(GeorgeWeb):
   * Dynamically determine the returned number of compute units, because
   * currently it's based hardcoded based on the hardware specifications of my
   * development machine (Inte CPU + Intel GPU).
   */
  if (queue.get_device().is_cpu()) {
    return 8;
  }
  // else: gpu
  return 16;
}

// determines the highest-possible-most-optimal size a work-group range length
template <int total_length>
static size_t best_work_group_length(const cl::sycl::queue &queue) {
  // max (pow of 2) compute units
  const size_t num_groups = roundup_cu(queue);
  // max work-group size
  const size_t group_size_limit =
      queue.get_device()
          .template get_info<cl::sycl::info::device::max_work_group_size>();
  return std::min(static_cast<size_t>(total_length / num_groups),
                  group_size_limit);
};

#endif  // SYCL_UTILS_SYCL_MEMORY_UTIL_HPP_

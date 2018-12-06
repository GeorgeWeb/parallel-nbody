#ifndef SYCL_UTILS_SYCL_MEMORY_UTIL_HPP_
#define SYCL_UTILS_SYCL_MEMORY_UTIL_HPP_

#include <CL/sycl.hpp>

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

#endif // SYCL_UTILS_SYCL_MEMORY_UTIL_HPP_

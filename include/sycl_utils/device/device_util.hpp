#ifndef SYCL_UTILS_SYCL_DEVICE_UTIL_HPP_
#define SYCL_UTILS_SYCL_DEVICE_UTIL_HPP_

#include <CL/sycl.hpp>

#include <iostream>

struct sycl_option_t {
  static constexpr int disabled = 0;
  static constexpr int enable_async_exceptions = 1 << 1;
  static constexpr int enable_profiling = 1 << 2;
  static constexpr int enable_all = enable_async_exceptions | enable_profiling;
};

/* @Description: constructs a SYCL queue for OpenCL/Host device.
 * Options can be set to:
 * enable catching asynchronous errors
 * enable profiling using the SYCL profiling API
 * enable both of the above.
 * By default any of the options are disabled. */
template <int option = sycl_option_t::disabled>
static cl::sycl::queue create_queue(
    const cl::sycl::device_selector &selected_device =
        cl::sycl::default_selector{}) {
  switch (option) {
    case sycl_option_t::disabled:
    default:
      return cl::sycl::queue{selected_device};
    case sycl_option_t::enable_profiling:
      return cl::sycl::queue(
          selected_device, cl::sycl::property_list{
                               cl::sycl::property::queue::enable_profiling()});
    case sycl_option_t::enable_async_exceptions:
      return cl::sycl::queue(
          selected_device, [&](cl::sycl::exception_list exceptions) {
            for (const auto &e : exceptions) {
              try {
                std::rethrow_exception(e);
              } catch (const cl::sycl::exception &e) {
                std::cerr << "Caught asynchronous SYCL exception:\n"
                          << e.what() << std::endl;
                // print the CL error code as well.
                const auto cl_error = e.get_cl_code();
                std::cerr << e.what() << "CL::ERROR::CODE : " << cl_error
                          << std::endl;
              }
            }
          });
    case sycl_option_t::enable_all:
      return cl::sycl::queue(
          selected_device,
          [&](cl::sycl::exception_list exceptions) {
            for (const auto &e : exceptions) {
              try {
                std::rethrow_exception(e);
              } catch (const cl::sycl::exception &e) {
                std::cerr << "Caught asynchronous SYCL exception:\n"
                          << e.what() << std::endl;
                // print the CL error code as well.
                const auto cl_error = e.get_cl_code();
                std::cerr << e.what() << "CL::ERROR::CODE : " << cl_error
                          << std::endl;
              }
            }
          },
          cl::sycl::property_list{
              cl::sycl::property::queue::enable_profiling()});
  }
}

#endif  // SYCL_UTILS_SYCL_DEVICE_UTIL_HPP_

#ifndef SYCL_UTILS_PROFILING_UTIL_HPP_
#define SYCL_UTILS_PROFILING_UTIL_HPP_

#include <CL/sycl.hpp>
#include <chrono>
#include <vector>

// container holding results of kernel profiling
template <typename data_t>
struct profiling_result {
  data_t total_kernel_submission_time;
  data_t total_kernel_execution_time;
  data_t total_application_execution_time;
  data_t total_application_execution_overhead_time;
};

/* @Description: helper function utilizing the SYCL profling API to do precise
 * kernel profiling */
template <typename data_t = double>
static profiling_result<data_t> sycl_profile(
    std::vector<cl::sycl::event> &events,
    const std::vector<std::chrono::time_point<std::chrono::system_clock>>
        &starts) {
  // time interval alias with generic precision type and nanoseconds format
  using time_interval_t = std::chrono::duration<data_t, std::micro>;

  // declare device time output variables
  auto total_kernel_submission_time = data_t{0.0};
  auto total_kernel_execution_time = data_t{0.0};
  // declare host time output variables
  auto total_application_execution_time = data_t{0.0};
  auto total_application_execution_overhead_time = data_t{0.0};
  // compute the total kernel submission and execution times
  const size_t events_size = events.size();
  for (size_t i = 0; i < events_size; i++) {
    // calculate host time
    events.at(i).wait();  // wait for all events to complete
    const auto end = std::chrono::system_clock::now();
    const time_interval_t current_application_execution_time = end - starts[i];
    total_application_execution_time +=
        current_application_execution_time.count();
    // calculate device time
    const auto cmd_submit_time =
        events.at(i)
            .template get_profiling_info<
                cl::sycl::info::event_profiling::command_submit>();  /// in ns
    const auto cmd_start_time =
        events.at(i)
            .template get_profiling_info<
                cl::sycl::info::event_profiling::command_start>();  /// in ns
    const auto cmd_end_time =
        events.at(i)
            .template get_profiling_info<
                cl::sycl::info::event_profiling::command_end>();  /// in ns
    constexpr auto ratio_denom = data_t{1000.0};  /// converter to μs
    // get kernel total time (from launch to end of execution)
    const auto current_submission_time =
        static_cast<data_t>((cmd_start_time - cmd_submit_time) / ratio_denom);
    const auto current_execution_time =
        static_cast<data_t>((cmd_end_time - cmd_start_time) / ratio_denom);
    total_kernel_submission_time += current_submission_time;
    total_kernel_execution_time += current_execution_time;
    // get host time overhead time
    const auto current_application_execution_overhead_time =
        total_application_execution_time -
        (total_kernel_submission_time + total_kernel_execution_time);
    total_application_execution_overhead_time +=
        current_application_execution_overhead_time;
  }

  // return a container struct holding all resulting time values
  return profiling_result<data_t>{total_kernel_submission_time,
                                  total_kernel_execution_time,
                                  total_application_execution_time,
                                  total_application_execution_overhead_time};
}

#endif  // SYCL_UTILS_PROFILING_UTIL_HPP_

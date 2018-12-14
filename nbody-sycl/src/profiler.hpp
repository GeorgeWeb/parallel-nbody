#ifndef NBODY_PROFILER_HPP_
#define NBODY_PROFILER_HPP_

// sycl helper abstractions and type aliases
#include <sycl_utils.hpp>

#include "file_io.hpp"

#include <chrono>
#include <string>
#include <vector>

#ifndef MAX_TIME_STEPS
#define MAX_TIME_STEPS 600
#endif  // MAX_TIME_STEPS

class Profiler {
 public:
  Profiler() = default;

  template <int AccessOption, typename data_t = double>
  inline void SyclToCSV(
      const std::string &name, std::vector<cl::sycl::event> &events,
      const std::vector<std::chrono::time_point<std::chrono::system_clock>>
          &starts) {
    // begin profiling
    if (static bool finished{false}; !finished) {
      // profile kernel submission and execution
      const auto profiling = sycl_profile<data_t>(events, starts);
      typedef file_io::FileIO fio;

      static const std::string mem_access_opt =
          AccessOption == data_access_t::coalesced
              ? "Coalesced"
              : AccessOption == data_access_t::local ? "Local" : "Global";
      static const std::string filename(name + mem_access_opt + ".csv");

      // display the current profiling output
      std::cout << "\n"
                << "kernel submission time: "
                << profiling.total_kernel_submission_time << " \u03BCs\n"
                << "kernel execution time: "
                << profiling.total_kernel_execution_time << " \u03BCs\n"
                << "application execution time : (real host time) "
                << profiling.total_application_execution_time << " \u03BCs\n"
                << "application execution overhead time: (host "
                   "overhead time) "
                << profiling.total_application_execution_overhead_time
                << " \u03BCs\n";

      // therefore add 1 more.
      // the first iteration includes building the program from the kernel
      static constexpr auto time_step_count_limit = MAX_TIME_STEPS + 1;
      // ...
      std::vector<profiling_result<data_t>> total_times;
      total_times.reserve(time_step_count_limit - 1);

      // create the csv file for time measurements (and add header row)
      if (gfx::time_step_count == 1) {
        // save header to file.
        const std::string header(
            "Memory Access Type, Average Kernel Submission Time, Average "
            "Kernel Execution Time, Average Host Overhead Time, Average "
            "Total (Real) Host Time");
        fio::instance().Save(header, filename);
        std::cout << std::endl << filename << std::endl << header << std::endl;
      }

      // populate the total times vector
      if (gfx::time_step_count >= 1 &&
          gfx::time_step_count < time_step_count_limit) {
        total_times.push_back(
            {profiling.total_kernel_submission_time,
             profiling.total_kernel_execution_time,
             profiling.total_application_execution_time,
             profiling.total_application_execution_overhead_time});
      }

      // get the average times from the total times vector (and save to file)
      if (gfx::time_step_count == time_step_count_limit - 1) {
        data_t avg_kernel_submission_time;
        data_t avg_kernel_execution_time;
        data_t avg_application_execution_time;
        data_t avg_application_execution_overhead_time;
        for (size_t i = 0; i < total_times.size(); i++) {
          avg_kernel_submission_time +=
              total_times.at(i).total_kernel_submission_time;
          avg_kernel_execution_time +=
              total_times.at(i).total_kernel_execution_time;
          avg_application_execution_time +=
              total_times.at(i).total_application_execution_time;
          avg_application_execution_overhead_time +=
              total_times.at(i).total_application_execution_overhead_time;
        }
        avg_kernel_submission_time /= total_times.size();
        avg_kernel_execution_time /= total_times.size();
        avg_application_execution_time /= total_times.size();
        avg_application_execution_overhead_time /= total_times.size();

        // display the average profiling output
        std::cout << "\nGlobal\n"
                  << "[AVERAGE] kernel submission time: "
                  << avg_kernel_submission_time << " \u03BCs\n"
                  << "[AVERAGE] kernel execution time: "
                  << avg_kernel_execution_time << " \u03BCs\n"
                  << "[AVERAGE] application execution time : (real host time) "
                  << avg_application_execution_time << " \u03BCs\n"
                  << "[AVERAGE] application execution overhead time: (host "
                     "overhead time) "
                  << avg_application_execution_overhead_time << " \u03BCs\n";

        // Save (append) measured data to file after the header.
        const auto body = std::string(
            mem_access_opt + ", " + std::to_string(avg_kernel_submission_time) +
            ", " + std::to_string(avg_kernel_execution_time) + ", " +
            std::to_string(avg_application_execution_overhead_time) + ", " +
            std::to_string(avg_application_execution_time));
        fio::instance().Save(body, filename);
        
        // end profiling
        std::cout << "Finished profling.\n";
        finished = true;

        // exit app.
        exit(1);
      }
    }
  }
};

#endif  // NBODY_FORMATTED_PROFILER_HPP_

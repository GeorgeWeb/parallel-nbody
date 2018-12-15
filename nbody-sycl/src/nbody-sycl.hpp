#ifndef NBODY_SYCL_HPP_
#define NBODY_SYCL_HPP_

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// cross-platform graphics library
#include <graphics.hpp>
namespace gfx = graphics;

// the nbody computation kernel functors
#include "kernels/compute_force_kernel.hpp"
#include "kernels/integrate_body_kernel.hpp"

// ...
#include "crand.hpp"

// ...
#include "profiler.hpp"

// ...
static constexpr auto sycl_option = sycl_option_t::disabled;
// how many submissions of each kernel will be executed
static constexpr auto submissions = 1u;

template <int num_bodies>
class NbodyScene {
 public:
  // graphics screen
  gfx::Window window;

  // constructs a scene from a window definition
  explicit NbodyScene(gfx::Window t_window) : window(std::move(t_window)) {
    // selecting device using sycl
    queue = create_queue<sycl_option>(sycl::gpu_selector{});
  }

  void OnLoad() {
    crand::SeedRand();

    // initialise shader program
    m_shader = std::make_unique<gfx::Shader>("shaders/default.vert",
                                             "shaders/default.frag");
    // initialise renderer for this scene
    m_renderer = std::make_unique<gfx::Renderer>();
    m_camera = std::make_unique<gfx::Camera>(glm::vec3(0.0f, 0.0f, 500.0f));

    // initialise the body system
    for (int i = 0; i < num_bodies; ++i) {
      // initialise body data
      m_positions.at(i) = {crand::GetRand(-200.0f, 200.0f),
                           crand::GetRand(-100.0f, 100.0f),
                           crand::GetRand(-50.0f, 50.0f)};
      m_gravities.at(i) = {0.0f, -9.8f, 0.0f};
      m_masses.at(i) = 1.0f;

      // initialise meshes
      m_meshes.at(i) = std::make_shared<gfx::Mesh>(gfx::Shape::CUBE);
    }

    // move meshes
    updateMeshPositions();
  }

  void OnUpdate() {
    // initialise the sycl buffers with the bodies data
    m_gravity_buf = std::make_shared<sycl::buffer<sycl::cl_float3, 1>>(
        m_gravities.data(), sycl::range<1>(num_bodies));

    m_velocity_buf = std::make_shared<sycl::buffer<sycl::cl_float3, 1>>(
        m_velocities.data(), sycl::range<1>(num_bodies));

    m_position_buf = std::make_shared<sycl::buffer<sycl::cl_float3, 1>>(
        m_positions.data(), sycl::range<1>(num_bodies));

    m_mass_buf = std::make_shared<sycl::buffer<float, 1>>(
        m_masses.data(), sycl::range<1>(num_bodies));

    // move bodies
    try {
      // submitting command-group that executes the ComputeForceKernel
      { computeForces<data_access_t::local>(); }
      // submitting command-group that executes the IntegrateBodyKernel
      { integrateBodies<data_access_t::global>(); }
    } catch (const sycl::exception &e) {
      std::cout << "Synchronous exception caught:\n" << e.what();
      exit(1);
    }

    // move meshes
    updateMeshPositions();
  }

  void OnDraw() {
    for (int i = 0; i < num_bodies; ++i) {
      m_renderer->Draw(m_shader, m_camera, m_meshes.at(i));
    }
  }

 private:
  // rendering controllers
  std::unique_ptr<gfx::Camera> m_camera;
  std::unique_ptr<gfx::Shader> m_shader;
  std::unique_ptr<gfx::Renderer> m_renderer;

  // meshes to graphically represent the body simulation
  std::array<std::shared_ptr<gfx::Mesh>, num_bodies> m_meshes;

  // basic physics body properties with SYCL underlying data types
  std::array<sycl::float3, num_bodies> m_gravities;
  std::array<sycl::float3, num_bodies> m_velocities;
  std::array<sycl::float3, num_bodies> m_positions;
  std::array<float, num_bodies> m_masses;

  // ...
  std::shared_ptr<sycl::buffer<sycl::cl_float3, 1>> m_gravity_buf;
  std::shared_ptr<sycl::buffer<sycl::cl_float3, 1>> m_velocity_buf;
  std::shared_ptr<sycl::buffer<sycl::cl_float3, 1>> m_position_buf;
  std::shared_ptr<sycl::buffer<float, 1>> m_mass_buf;

  // OpenCL SYCL queue; used to select device and submit command groups
  sycl::queue queue;

  /* computes gravity force (optimised) with several kernel implementations
   * allowing to choose from: global, local and coalesced - OpenCL SYCL device
   * memory access
   */
  template <int AccessOption>
  inline void computeForces() {
    switch (AccessOption) {
      case data_access_t::global: {
        // initialize a vector of sycl event objects to save the profiling data
        std::vector<sycl::event> events(submissions);
        // initialize a vector of time points save each start of the computation
        std::vector<std::chrono::time_point<std::chrono::system_clock>> starts(
            submissions);

        // submit the command-group and execute the kernel
        for (size_t i = 0; i < submissions; ++i) {
          starts.at(i) = std::chrono::system_clock::now();
          events.at(i) = queue.submit([&](sycl::handler &cgh) {
            // define global accessors
            auto position_ptr = m_position_buf->get_access<sycl_mode_read>(cgh);
            auto mass_ptr = m_mass_buf->get_access<sycl_mode_read>(cgh);
            auto gravity_ptr = m_gravity_buf->get_access<sycl_mode_write>(cgh);

            // setup the work group sizes and run the kernel
            cgh.parallel_for(
                // setup the range
                sycl::range<1>(num_bodies),
                // call the kernel functor
                kernels::ComputeForceKernel<data_access_t::global, num_bodies>(
                    position_ptr, mass_ptr, gravity_ptr));
          });
        }
        // profile time complexity of the entire execution
        if (sycl_option == sycl_option_t::enable_all ||
            sycl_option == sycl_option_t::enable_profiling) {
          Profiler profiler{};
          profiler.SyclToCSV<data_access_t::global>(
              "cpu/ComputeForceKernel", events, starts);
        }
      } break;
      case data_access_t::local: {
        // initialize a vector of sycl event objects to save the profiling data
        std::vector<sycl::event> events(submissions);
        // initialize a vector of time points save each start of the computation
        std::vector<std::chrono::time_point<std::chrono::system_clock>> starts(
            submissions);

        // submit the command-group and execute the kernel
        for (size_t i = 0; i < submissions; ++i) {
          starts.at(i) = std::chrono::system_clock::now();
          events.at(i) = queue.submit([&](sycl::handler &cgh) {
            // define global accessors
            auto position_ptr = m_position_buf->get_access<sycl_mode_read>(cgh);
            auto mass_ptr = m_mass_buf->get_access<sycl_mode_read>(cgh);
            auto gravity_ptr = m_gravity_buf->get_access<sycl_mode_write>(cgh);

            // define local accessors
            const auto local_size = best_work_group_length<num_bodies>(queue);
            read_write_accessor_t<sycl::float3, 1, sycl_target_local>
                gravity_scratch_ptr(local_size, cgh);

            // setup the work group sizes and run the kernel
            cgh.parallel_for(
                // setup the range
                sycl::nd_range<1>(sycl::range<1>(num_bodies),
                                  sycl::range<1>(local_size)),
                // call the kernel functor
                kernels::ComputeForceKernel<data_access_t::local, num_bodies>(
                    gravity_scratch_ptr, position_ptr, mass_ptr, gravity_ptr));
          });
        }
        // profile time complexity of the entire execution
        if (sycl_option == sycl_option_t::enable_all ||
            sycl_option == sycl_option_t::enable_profiling) {
          Profiler profiler{};
          profiler.SyclToCSV<data_access_t::local>(
              "cpu/ComputeForceKernel", events, starts);
        }
      } break;
      case data_access_t::coalesced:
        // it is not that much of a memory-bound problem like the
        // integration
        break;
      default:
        break;
    }
  }

  /* computes body velocity and position integration with several kernel
   * implementations allowing to choose from: global, local and coalesced -
   * OpenCL SYCL device memory access.
   */
  template <int AccessOption>
  inline void integrateBodies() {
    switch (AccessOption) {
      case data_access_t::global: {
        // initialize a vector of sycl event objects to save the profiling data
        std::vector<sycl::event> events(submissions);
        // initialize a vector of time points save each start of the computation
        std::vector<std::chrono::time_point<std::chrono::system_clock>> starts(
            submissions);
        // submit the command-group and execute the kernel
        for (size_t i = 0; i < submissions; ++i) {
          starts.at(i) = std::chrono::system_clock::now();
          events.at(i) = queue.submit([&](sycl::handler &cgh) {
            // define global accessors
            const auto gravity_ptr =
                m_gravity_buf->get_access<sycl_mode_read>(cgh);
            auto velocity_ptr =
                m_velocity_buf->get_access<sycl_mode_read_write>(cgh);
            auto position_ptr =
                m_position_buf->get_access<sycl_mode_read_write>(cgh);

            // setup the work group sizes and run the kernel
            cgh.parallel_for(
                // setup the range
                sycl::range<1>(num_bodies),
                // call the kernel functor
                kernels::IntegrateBodyKernel<data_access_t::global, num_bodies>(
                    gravity_ptr, velocity_ptr, position_ptr, gfx::delta_time));
          });
        }
        // profile time complexity of the entire execution
        if (sycl_option == sycl_option_t::enable_all ||
            sycl_option == sycl_option_t::enable_profiling) {
          Profiler profiler{};
          profiler.SyclToCSV<data_access_t::global>(
              "cpu/IntegrateBodyKernel", events, starts);
        }
      } break;
      case data_access_t::local: {
        // initialize a vector of sycl event objects to save the profiling data
        std::vector<sycl::event> events(submissions);
        // initialize a vector of time points save each start of the computation
        std::vector<std::chrono::time_point<std::chrono::system_clock>> starts(
            submissions);

        // submit the command-group and execute the kernel
        for (size_t i = 0; i < submissions; ++i) {
          starts.at(i) = std::chrono::system_clock::now();
          events.at(i) = queue.submit([&](sycl::handler &cgh) {
            // define global accessors
            const auto gravity_ptr =
                m_gravity_buf->get_access<sycl_mode_read>(cgh);
            auto velocity_ptr =
                m_velocity_buf->get_access<sycl_mode_read_write>(cgh);
            auto position_ptr =
                m_position_buf->get_access<sycl_mode_read_write>(cgh);

            // define local accessors
            const auto local_size = best_work_group_length<num_bodies>(queue);
            read_write_accessor_t<sycl::float3, 1, sycl_target_local>
                velocity_scratch_ptr(local_size, cgh);
            read_write_accessor_t<sycl::float3, 1, sycl_target_local>
                position_scratch_ptr(local_size, cgh);

            // setup the work group sizes and run the kernel
            cgh.parallel_for(
                // setup the range
                sycl::nd_range<1>(sycl::range<1>(num_bodies),
                                  sycl::range<1>(local_size)),
                // call the kernel functor
                kernels::IntegrateBodyKernel<data_access_t::local, num_bodies>(
                    velocity_scratch_ptr, position_scratch_ptr, gravity_ptr,
                    velocity_ptr, position_ptr, gfx::delta_time));
          });
        }
        // profile time complexity of the entire execution
        if (sycl_option == sycl_option_t::enable_all ||
            sycl_option == sycl_option_t::enable_profiling) {
          Profiler profiler{};
          profiler.SyclToCSV<data_access_t::local>(
              "cpu/IntegrateBodyKernel", events, starts);
        }
      } break;
      case data_access_t::coalesced: {
        // initialize a vector of sycl event objects to save the profiling data
        std::vector<sycl::event> events(submissions);
        // initialize a vector of time points save each start of the computation
        std::vector<std::chrono::time_point<std::chrono::system_clock>> starts(
            submissions);

        // submit the command-group and execute the kernel
        for (size_t i = 0; i < submissions; ++i) {
          starts.at(i) = std::chrono::system_clock::now();
          events.at(i) = queue.submit([&](sycl::handler &cgh) {
            // define global accessors
            const auto gravity_ptr =
                m_gravity_buf->get_access<sycl_mode_read>(cgh);
            auto velocity_ptr =
                m_velocity_buf->get_access<sycl_mode_read_write>(cgh);
            auto position_ptr =
                m_position_buf->get_access<sycl_mode_read_write>(cgh);

            // setup the work group sizes and run the kernel
            cgh.parallel_for(
                // setup the range
                sycl::nd_range<1>(sycl::range<1>(num_bodies),
                                  sycl::range<1>(opencl_config::cache_line)),
                // call the kernel functor
                kernels::IntegrateBodyKernel<data_access_t::coalesced,
                                             num_bodies>(
                    gravity_ptr, velocity_ptr, position_ptr, gfx::delta_time));
          });
        }
        // profile time complexity of the entire execution
        if (sycl_option == sycl_option_t::enable_all ||
            sycl_option == sycl_option_t::enable_profiling) {
          Profiler profiler{};
          profiler.SyclToCSV<data_access_t::coalesced>(
              "cpu/IntegrateBodyKernel", events, starts);
        }
      } break;
      default:
        break;
    }
  }

  // sets mesh m_positions to match the newly computed body position
  inline void updateMeshPositions() {
    for (int i = 0; i < num_bodies; ++i) {
      m_meshes.at(i)->SetPosition(glm::vec3(
          m_positions.at(i).x(), m_positions.at(i).y(), m_positions.at(i).z()));
    }
  }
};

#endif  // NBODY_SYCL_HPP_

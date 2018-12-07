#ifndef NBODY_SYCL_HPP_
#define NBODY_SYCL_HPP_

#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <memory>
#include <random>
#include <string>

// cross-platform graphics library
#include <graphics.hpp>
namespace gfx = graphics;

// the nbody computation kernel functors
#include "kernels/compute_force_kernel.hpp"
#include "kernels/integrate_body_kernel.hpp"

// abstract in a header file in the root /include dir.
static float GetRand(float lo, float hi) {
  return lo + static_cast<float>(rand()) /
                  (static_cast<float>(RAND_MAX / (hi - lo)));
}

template <int num_bodies>
class NbodyScene {
 public:
  // graphics screen
  gfx::Window window;

  // constructs a scene from a window definition
  explicit NbodyScene(gfx::Window t_window) : window(std::move(t_window)) {
    // selecting device using sycl
    queue = create_queue(sycl::cpu_selector{});
  }

  void OnLoad() {
    // c-style seeding the rand() generator
    srand(time(0));

    // initialise m_shader program
    m_shader = std::make_unique<gfx::Shader>("shaders/default.vert",
                                             "shaders/default.frag");
    // initialise m_renderer for this scene
    m_renderer = std::make_unique<gfx::Renderer>();
    m_camera = std::make_unique<gfx::Camera>(glm::vec3(0.0f, 0.0f, 500.0f));

    // initialise the body system
    for (int i = 0; i < num_bodies; ++i) {
      // initialise body data
      m_positions.at(i) = {GetRand(-200.0f, 200.0f), GetRand(-100.0f, 100.0f),
                           GetRand(-50.0f, 50.0f)};
      m_gravities.at(i) = {0.0f, -9.8f, 0.0f};
      m_masses.at(i) = 1.0f;

      // initialise m_meshes
      m_meshes.at(i) = std::make_shared<gfx::Mesh>(gfx::Shape::CUBE);
    }

    // move m_meshes
    updateMeshPositions();
  }

  void OnUpdate() {
    // move bodies
    if (queue.get_device().is_cpu()) {
      computeForces<data_access_t::coalesced>();
      integrateBodies<data_access_t::global>();
    } else if (queue.get_device().is_gpu()) {
      computeForces<data_access_t::coalesced>();
      integrateBodies<data_access_t::local>();
    } else {
      std::cout << "\nApplication is shutting down ...\n";
      std::cout << "The selected device is not considered for this program\n\n";
      // note: for the same reasons the SYCL host device is also avoided
      exit(1);
    }

    // move m_meshes
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

  // m_meshes to graphically represent the body simulation
  std::array<std::shared_ptr<gfx::Mesh>, num_bodies> m_meshes;

  // basic physics body properties with SYCL underlying data types
  std::array<cl::sycl::float3, num_bodies> m_gravities;
  std::array<cl::sycl::float3, num_bodies> m_velocities;
  std::array<cl::sycl::float3, num_bodies> m_positions;
  std::array<float, num_bodies> m_masses;

  // OpenCL SYCL queue; used to select device and submit command groups
  sycl::queue queue;

  /* computes gravity force (optimised) with several kernel implementations
   * allowing to choose from: global, local and coalesced - OpenCL SYCL device
   * memory access
   */
  template <int AccessOption>
  inline void computeForces() {
    switch (AccessOption) {
      case data_access_t::global:
        try {
          // create buffers from the host data
          sycl::buffer<cl::sycl::cl_float3, 1> position_buf(
              m_positions.data(), sycl::range<1>(num_bodies));
          sycl::buffer<float, 1> mass_buf(m_masses.data(),
                                          sycl::range<1>(num_bodies));
          sycl::buffer<cl::sycl::cl_float3, 1> gravity_buf(
              m_gravities.data(), sycl::range<1>(num_bodies));

          // submit the command group
          queue.submit([&](sycl::handler &cgh) {
            // define global accessors
            auto position_ptr = position_buf.get_access<sycl_mode_read>(cgh);
            auto mass_ptr = mass_buf.get_access<sycl_mode_read>(cgh);
            auto gravity_ptr = gravity_buf.get_access<sycl_mode_write>(cgh);

            // setup the work group sizes and run the kernel
            cgh.parallel_for(
                // setup the range
                sycl::range<1>(num_bodies),
                // call the kernel functor
                kernels::ComputeForceKernel<data_access_t::global, num_bodies>(
                    position_ptr, mass_ptr, gravity_ptr));
          });
        } catch (const sycl::exception &e) {
          std::cout << "Synchronous exception caught:\n" << e.what();
          exit(1);
        }
        break;
      case data_access_t::local:
        try {
          // create buffers from the host data
          sycl::buffer<cl::sycl::cl_float3, 1> position_buf(
              m_positions.data(), sycl::range<1>(num_bodies));
          sycl::buffer<float, 1> mass_buf(m_masses.data(),
                                          sycl::range<1>(num_bodies));
          sycl::buffer<cl::sycl::cl_float3, 1> gravity_buf(
              m_gravities.data(), sycl::range<1>(num_bodies));

          // submit the command group
          queue.submit([&](sycl::handler &cgh) {
            // define global accessors
            auto position_ptr = position_buf.get_access<sycl_mode_read>(cgh);
            auto mass_ptr = mass_buf.get_access<sycl_mode_read>(cgh);
            auto gravity_ptr = gravity_buf.get_access<sycl_mode_write>(cgh);

            // define local accessors
            const auto local_size = best_work_group_length<num_bodies>(queue);
            read_write_accessor_t<cl::sycl::float3, 1, sycl_target_local>
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
        } catch (const sycl::exception &e) {
          std::cout << "Synchronous exception caught:\n" << e.what();
          exit(1);
        }
        break;
      case data_access_t::coalesced:
        try {
          // create buffers from the host data
          sycl::buffer<cl::sycl::cl_float3, 1> position_buf(
              m_positions.data(), sycl::range<1>(num_bodies));
          sycl::buffer<float, 1> mass_buf(m_masses.data(),
                                          sycl::range<1>(num_bodies));
          sycl::buffer<cl::sycl::cl_float3, 1> gravity_buf(
              m_gravities.data(), sycl::range<1>(num_bodies));

          // define local accessors
          const auto num_groups = roundup_cu(queue);
          // getting the maximum work group size per thread
          const auto work_group_size =
              best_work_group_length<num_bodies>(queue);  // 256;
          // building the best number of global thread
          const auto total_threads = num_groups * work_group_size;

          // submit the command group
          queue.submit([&](sycl::handler &cgh) {
            // define global accessors
            auto position_ptr = position_buf.get_access<sycl_mode_read>(cgh);
            auto mass_ptr = mass_buf.get_access<sycl_mode_read>(cgh);
            auto gravity_ptr = gravity_buf.get_access<sycl_mode_write>(cgh);

            // setup the work group sizes and run the kernel
            cgh.parallel_for(
                // setup the range
                sycl::nd_range<1>(sycl::range<1>(total_threads / 2),
                                  sycl::range<1>(work_group_size / 2)),
                // call the kernel functor
                kernels::ComputeForceKernel<data_access_t::coalesced,
                                            num_bodies>(position_ptr, mass_ptr,
                                                        gravity_ptr));
          });
        } catch (const sycl::exception &e) {
          std::cout << "Synchronous exception caught:\n" << e.what();
          exit(1);
        }
        break;
      default:
        // must be called with one of the above parameters
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
      case data_access_t::global:
        try {
          // create buffers from the host data
          sycl::buffer<cl::sycl::cl_float3, 1> gravity_buf(
              m_gravities.data(), sycl::range<1>(num_bodies));
          sycl::buffer<cl::sycl::cl_float3, 1> veclocity_buf(
              m_velocities.data(), sycl::range<1>(num_bodies));
          sycl::buffer<cl::sycl::cl_float3, 1> position_buf(
              m_positions.data(), sycl::range<1>(num_bodies));

          // submit the command group
          queue.submit([&](sycl::handler &cgh) {
            // define global accessors
            const auto gravity_ptr =
                gravity_buf.get_access<sycl_mode_read>(cgh);
            auto velocity_ptr =
                veclocity_buf.get_access<sycl_mode_read_write>(cgh);
            auto position_ptr =
                position_buf.get_access<sycl_mode_read_write>(cgh);

            // setup the work group sizes and run the kernel
            cgh.parallel_for(
                // setup the range
                sycl::range<1>(num_bodies),
                // call the kernel functor
                kernels::IntegrateBodyKernel<data_access_t::global, num_bodies>(
                    gravity_ptr, velocity_ptr, position_ptr, gfx::delta_time));
          });
        } catch (const sycl::exception &e) {
          std::cout << "Synchronous exception caught:\n" << e.what();
          exit(1);
        }
        break;
      case data_access_t::local:
        try {
          // create buffers from the host data
          sycl::buffer<cl::sycl::cl_float3, 1> gravity_buf(
              m_gravities.data(), sycl::range<1>(num_bodies));
          sycl::buffer<cl::sycl::cl_float3, 1> veclocity_buf(
              m_velocities.data(), sycl::range<1>(num_bodies));
          sycl::buffer<cl::sycl::cl_float3, 1> position_buf(
              m_positions.data(), sycl::range<1>(num_bodies));

          // submit the command group
          queue.submit([&](sycl::handler &cgh) {
            // define global accessors
            const auto gravity_ptr =
                gravity_buf.get_access<sycl_mode_read>(cgh);
            auto velocity_ptr =
                veclocity_buf.get_access<sycl_mode_read_write>(cgh);
            auto position_ptr =
                position_buf.get_access<sycl_mode_read_write>(cgh);

            // define local accessors
            const auto local_size = best_work_group_length<num_bodies>(queue);
            read_write_accessor_t<cl::sycl::float3, 1, sycl_target_local>
                velocity_scratch_ptr(local_size, cgh);
            read_write_accessor_t<cl::sycl::float3, 1, sycl_target_local>
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
        } catch (const sycl::exception &e) {
          std::cout << "Synchronous exception caught:\n" << e.what();
          exit(1);
        }
        break;
      case data_access_t::coalesced:
        // TODO: ...
        break;
      default:
        // must be called with one of the above parameters
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

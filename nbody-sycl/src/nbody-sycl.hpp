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
  // replace these 3 with a window instance
  int width;
  int height;
  std::string title;

  std::array<std::shared_ptr<gfx::Mesh>, num_bodies> meshes;

  std::unique_ptr<gfx::Camera> camera;
  std::unique_ptr<gfx::Shader> shader;
  std::unique_ptr<gfx::Renderer> renderer;

  NbodyScene(int t_width, int t_height, std::string_view t_title)
      : width(t_width), height(t_height), title(t_title) {
    // selecting device using sycl
    queue = create_queue();
  }

  void OnLoad() {
    // c-style seeding the rand() generator
    srand(time(0));

    // initialise shader program
    shader = std::make_unique<gfx::Shader>("shaders/default.vert",
                                           "shaders/default.frag");
    // initialise renderer for this scene
    renderer = std::make_unique<gfx::Renderer>();
    camera = std::make_unique<gfx::Camera>(glm::vec3(0.0f, 0.0f, 500.0f));

    // initialise the body system
    for (int i = 0; i < num_bodies; ++i) {
      // initialise body data
      positions.at(i) = {GetRand(-200.0f, 200.0f),
                         GetRand(-100.0f, 100.0f),
                         GetRand(-50.0f, 50.0f)};
      gravities.at(i) = {0.0f, -9.8f, 0.0f};
      masses.at(i) = 1.0f;

      // initialise meshes
      meshes.at(i) = std::make_shared<gfx::Mesh>(gfx::Shape::CUBE);
    }

    // move meshes
    updateMeshPositions();
  }

  void OnUpdate() {
    // move bodies
    computeForces<data_access_t::global>();
    integrateBodies<data_access_t::local>();

    // move meshes
    updateMeshPositions();
  }

  void OnDraw() {
    for (int i = 0; i < num_bodies; ++i) {
      renderer->Draw(shader, camera, meshes.at(i));
    }
  }

 private:
  sycl::queue queue;

  std::array<cl::sycl::float3, num_bodies> gravities;
  std::array<cl::sycl::float3, num_bodies> velocities;
  std::array<cl::sycl::float3, num_bodies> positions;
  std::array<float, num_bodies> masses;

  template <int access>
  inline void computeForces() {
    switch (access) {
      case data_access_t::global:
        try {
          sycl::buffer<cl::sycl::cl_float3, 1> position_buf(
              positions.data(), sycl::range<1>(num_bodies));
          sycl::buffer<float, 1> mass_buf(masses.data(),
                                          sycl::range<1>(num_bodies));
          sycl::buffer<cl::sycl::cl_float3, 1> gravity_buf(
              gravities.data(), sycl::range<1>(num_bodies));
          queue.submit([&](sycl::handler &cgh) {
            auto position_ptr = position_buf.get_access<sycl_mode_read>(cgh);
            auto mass_ptr = mass_buf.get_access<sycl_mode_read>(cgh);
            auto gravity_ptr = gravity_buf.get_access<sycl_mode_write>(cgh);
            cgh.parallel_for(
                sycl::range<1>(num_bodies),
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
          sycl::buffer<cl::sycl::cl_float3, 1> position_buf(
              positions.data(), sycl::range<1>(num_bodies));
          sycl::buffer<float, 1> mass_buf(masses.data(),
                                          sycl::range<1>(num_bodies));
          sycl::buffer<cl::sycl::cl_float3, 1> gravity_buf(
              gravities.data(), sycl::range<1>(num_bodies));
          queue.submit([&](sycl::handler &cgh) {
            auto position_ptr = position_buf.get_access<sycl_mode_read>(cgh);
            auto mass_ptr = mass_buf.get_access<sycl_mode_read>(cgh);
            auto gravity_ptr = gravity_buf.get_access<sycl_mode_write>(cgh);
            cgh.parallel_for(
                sycl::range<1>(num_bodies),
                kernels::ComputeForceKernel<data_access_t::local, num_bodies>(
                    position_ptr, mass_ptr, gravity_ptr));
          });
        } catch (const sycl::exception &e) {
          std::cout << "Synchronous exception caught:\n" << e.what();
          exit(1);
        }
        break;
      default:
        break;
    }
  }

  template <int access>
  inline void integrateBodies() {
    switch (access) {
      case data_access_t::global:
        try {
          sycl::buffer<cl::sycl::cl_float3, 1> gravity_buf(
              gravities.data(), sycl::range<1>(num_bodies));
          sycl::buffer<cl::sycl::cl_float3, 1> veclocity_buf(
              velocities.data(), sycl::range<1>(num_bodies));
          sycl::buffer<cl::sycl::cl_float3, 1> position_buf(
              positions.data(), sycl::range<1>(num_bodies));

          queue.submit([&](sycl::handler &cgh) {
            const auto gravity_ptr =
                gravity_buf.get_access<sycl_mode_read>(cgh);
            auto velocity_ptr =
                veclocity_buf.get_access<sycl_mode_read_write>(cgh);
            auto position_ptr =
                position_buf.get_access<sycl_mode_read_write>(cgh);
            cgh.parallel_for(
                sycl::range<1>(num_bodies),
                kernels::IntegrateBodyKernel<data_access_t::global, num_bodies>(
                    gravity_ptr, velocity_ptr, position_ptr, gfx::delta_time));
          });
        } catch (const sycl::exception &e) {
          std::cout << "Synchronous exception caught:\n" << e.what();
          exit(1);
        }
        break;
      case data_access_t::local:
        // find a suitable local group score (for GPUs)
        static const auto get_optimal_local = [&]() -> size_t {
          const size_t work_group_size_limit =
              queue.get_device()
                  .template get_info<sycl::info::device::max_work_group_size>();
          // return the minimum from max work-group size and total num of
          // elements
          return std::min(static_cast<size_t>(num_bodies),
                          work_group_size_limit);
        };
        // ...
        try {
          sycl::buffer<cl::sycl::cl_float3, 1> gravity_buf(
              gravities.data(), sycl::range<1>(num_bodies));
          sycl::buffer<cl::sycl::cl_float3, 1> veclocity_buf(
              velocities.data(), sycl::range<1>(num_bodies));
          sycl::buffer<cl::sycl::cl_float3, 1> position_buf(
              positions.data(), sycl::range<1>(num_bodies));

          // submit the command group
          queue.submit([&](sycl::handler &cgh) {
            const auto gravity_ptr =
                gravity_buf.get_access<sycl_mode_read>(cgh);
            auto velocity_ptr =
                veclocity_buf.get_access<sycl_mode_read_write>(cgh);
            auto position_ptr =
                position_buf.get_access<sycl_mode_read_write>(cgh);

            // scratch-pads
            read_write_accessor_t<cl::sycl::float3, 1, sycl_target_local>
                velocity_scratch_ptr(get_optimal_local(), cgh);
            read_write_accessor_t<cl::sycl::float3, 1, sycl_target_local>
                position_scratch_ptr(get_optimal_local(), cgh);

            // setup the work group sizes and run the kernel
            cgh.parallel_for(
                // range
                sycl::nd_range<1>(sycl::range<1>(num_bodies),
                                  sycl::range<1>(get_optimal_local())),
                // kernel functor
                kernels::IntegrateBodyKernel<data_access_t::local, num_bodies>(
                    velocity_scratch_ptr, position_scratch_ptr, gravity_ptr,
                    velocity_ptr, position_ptr, gfx::delta_time));
          });
        } catch (const sycl::exception &e) {
          std::cout << "Synchronous exception caught:\n" << e.what();
          exit(1);
        }
        break;
      default:
        break;
    }
  }

  inline void updateMeshPositions() {
    for (int i = 0; i < num_bodies; ++i) {
      meshes.at(i)->SetPosition(glm::vec3(
          positions.at(i).x(), positions.at(i).y(), positions.at(i).z()));
    }
  }
};

#endif  // NBODY_SYCL_HPP_

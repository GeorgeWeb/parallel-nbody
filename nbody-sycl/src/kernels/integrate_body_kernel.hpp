#ifndef NBODY_SYCL_INTEGRATE_BODY_KERNEL_HPP_
#define NBODY_SYCL_INTEGRATE_BODY_KERNEL_HPP_

// ...
#include "common.hpp"

// sycl helper abstractions and type aliases
#include <sycl_utils.hpp>
namespace sycl = cl::sycl;

namespace kernels {

template <int access, int num_bodies>
class IntegrateBodyKernel;

// global memory access partial specialisation definition
template <int num_bodies>
class IntegrateBodyKernel<data_access_t::global, num_bodies> {
 public:
  // initialises the kernel data
  IntegrateBodyKernel(const read_accessor_t<sycl::float3, 1> gravity_ptr,
                      read_write_accessor_t<sycl::float3, 1> velocity_ptr,
                      read_write_accessor_t<sycl::float3, 1> position_ptr,
                      const float time_step)
      : m_gravity_ptr(gravity_ptr),
        m_velocity_ptr(velocity_ptr),
        m_position_ptr(position_ptr),
        m_time_step(time_step) {}

  // defines the kernel computation
  void operator()(sycl::item<1> item) {
    const auto id = item.get_id();
    m_velocity_ptr[id] += m_gravity_ptr[id] * m_time_step;
    m_position_ptr[id] += m_velocity_ptr[id] * m_time_step;
  }

 private:
  // global data accessors
  const read_accessor_t<sycl::float3, 1> m_gravity_ptr;
  read_write_accessor_t<sycl::float3, 1> m_velocity_ptr;
  read_write_accessor_t<sycl::float3, 1> m_position_ptr;

  // simulation time step
  const float m_time_step;
};

// local memory access partial specialisation definition
template <int num_bodies>
class IntegrateBodyKernel<data_access_t::local, num_bodies> {
 public:
  // initialises the kernel data
  IntegrateBodyKernel(
      read_write_accessor_t<sycl::float3, 1, sycl::access::target::local>
          velocity_scratch_ptr,
      read_write_accessor_t<sycl::float3, 1, sycl::access::target::local>
          position_scratch_ptr,
      const read_accessor_t<sycl::float3, 1> gravity_ptr,
      read_write_accessor_t<sycl::float3, 1> velocity_ptr,
      read_write_accessor_t<sycl::float3, 1> position_ptr,
      const float time_step)
      : m_velocity_scratch_ptr(velocity_scratch_ptr),
        m_position_scratch_ptr(position_scratch_ptr),
        m_gravity_ptr(gravity_ptr),
        m_velocity_ptr(velocity_ptr),
        m_position_ptr(position_ptr),
        m_time_step(time_step) {}

  // defines the kernel computation
  void operator()(sycl::nd_item<1> item) {
    const size_t global_id = item.get_global_id(0);
    const size_t local_id = item.get_local_id(0);

    // read to data local device memory
    m_velocity_scratch_ptr[local_id] = m_velocity_ptr[global_id];
    m_position_scratch_ptr[local_id] = m_position_ptr[global_id];

    // sync work-items within the work-group when writing is over
    item.barrier(sycl::access::fence_space::local_space);

    // integrate the new velocities and positions
    m_velocity_scratch_ptr[local_id] += m_gravity_ptr[global_id] * m_time_step;
    m_position_scratch_ptr[local_id] +=
        m_velocity_scratch_ptr[local_id] * m_time_step;

    // sync work-items within the work-group when writing is over
    item.barrier(sycl::access::fence_space::local_space);

    if (global_id < num_bodies) {
      // output the calculated results from the local device memory
      m_velocity_ptr[global_id] = m_velocity_scratch_ptr[local_id];
      m_position_ptr[global_id] = m_position_scratch_ptr[local_id];
    }
  }

 private:
  // local (scratch-pad) accessors
  read_write_accessor_t<sycl::float3, 1, sycl::access::target::local>
      m_velocity_scratch_ptr;
  read_write_accessor_t<sycl::float3, 1, sycl::access::target::local>
      m_position_scratch_ptr;
  // global (and constant) data accessors
  const read_accessor_t<sycl::float3, 1> m_gravity_ptr;
  read_write_accessor_t<sycl::float3, 1> m_velocity_ptr;
  read_write_accessor_t<sycl::float3, 1> m_position_ptr;

  // simulation time step
  const float m_time_step;
};

// global memory access partial specialisation definition
template <int num_bodies>
class IntegrateBodyKernel<data_access_t::coalesced, num_bodies> {
 public:
  // initialises the kernel data
  IntegrateBodyKernel(const read_accessor_t<sycl::float3, 1> gravity_ptr,
                      read_write_accessor_t<sycl::float3, 1> velocity_ptr,
                      read_write_accessor_t<sycl::float3, 1> position_ptr,
                      const float time_step)
      : m_gravity_ptr(gravity_ptr),
        m_velocity_ptr(velocity_ptr),
        m_position_ptr(position_ptr),
        m_time_step(time_step) {}

  // defines the kernel computation
  void operator()(sycl::nd_item<1> item) {
    const size_t global_id = item.get_global_id(0) * opencl_config::cache_line;
    const size_t group_size = m_gravity_ptr.get_count();

    sycl::float3 private_gravity[opencl_config::cache_line];
    sycl::float3 private_velocity[opencl_config::cache_line];
    sycl::float3 private_position[opencl_config::cache_line];

    for (size_t i = global_id; i < group_size;
         i += (item.get_global_range()[0] * opencl_config::cache_line)) {
      for (size_t k = 0; k < opencl_config::cache_line;
           k += opencl_config::cache_line) {
        // read in gravity
        for (size_t j = 0; j < opencl_config::cache_line; ++j) {
          private_gravity[j] = m_gravity_ptr[k + i + j];
        }
        // synchronise
        item.mem_fence(sycl::access::fence_space::global_and_local);
        // read in velocity
        for (size_t j = 0; j < opencl_config::cache_line; ++j) {
          private_velocity[j] = m_velocity_ptr[k + i + j];
        }
        // synchronise
        item.mem_fence(sycl::access::fence_space::global_and_local);
        // read in position
        for (size_t j = 0; j < opencl_config::cache_line; ++j) {
          private_position[j] = m_position_ptr[k + i + j];
        }
        // synchronise
        item.mem_fence(sycl::access::fence_space::global_and_local);
        // compute velocity
        for (size_t j = 0; j < opencl_config::cache_line; ++j) {
          private_velocity[j] += private_gravity[j] * m_time_step;
        }
        // synchronise
        item.mem_fence(sycl::access::fence_space::global_and_local);
        // compute position
        for (size_t j = 0; j < opencl_config::cache_line; ++j) {
          private_position[j] += private_velocity[j] * m_time_step;
        }
        // synchronise
        item.mem_fence(sycl::access::fence_space::global_and_local);
        // output velocity
        for (size_t j = 0; j < opencl_config::cache_line; ++j) {
          m_velocity_ptr[k + i + j] = private_velocity[j];
        }
        // synchronise
        item.mem_fence(sycl::access::fence_space::global_and_local);
        // output position
        for (size_t j = 0; j < opencl_config::cache_line; ++j) {
          m_position_ptr[k + i + j] = private_position[j];
        }
        // synchronise
        item.mem_fence(sycl::access::fence_space::global_and_local);
      }
    }
  }

 private:
  // global data accessors
  const read_accessor_t<sycl::float3, 1> m_gravity_ptr;
  read_write_accessor_t<sycl::float3, 1> m_velocity_ptr;
  read_write_accessor_t<sycl::float3, 1> m_position_ptr;

  // simulation time step
  const float m_time_step;
};

}  // namespace kernels

#endif  // NBODY_SYCL_INTEGRATE_BODY_KERNEL_HPP_

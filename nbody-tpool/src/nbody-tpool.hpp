#ifndef NBODY_TPOOL_HPP_
#define NBODY_TPOOL_HPP_

#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "tpool/tpool.hpp"

// cross-platform graphics library
#include <graphics.hpp>
namespace gfx = graphics;

#include "crand.hpp"
#include "file_io.hpp"
#include "timer.hpp"

#ifndef MAX_TIME_STEPS
#define MAX_TIME_STEPS 10
#endif  // MAX_TIME_STEPS

// gravity constant
static constexpr float k_grav = 6.67408f;

// ...
static const auto thread_num = std::thread::hardware_concurrency();

class Body {
 public:
  // default mesh setup constructor resulting in graphics representation is
  // limited to cube meshes at this stage
  Body()
      : m_mesh(std::make_shared<gfx::Mesh>(gfx::Shape::CUBE)),
        m_mass(1.0f),
        m_gravity(glm::vec3(0.0f, -9.8f, 0.0f)) {}

  std::shared_ptr<gfx::Mesh> GetMesh() const { return m_mesh; }

  glm::vec3 GetPosition() const { return m_position; }
  glm::vec3 GetVelocity() const { return m_velocity; }
  glm::vec3 GetAcceleration() const { return m_acceleration; }
  glm::vec3 GetGravity() const { return m_gravity; }
  float GetMass() const { return m_mass; }

  void SetPosition(const glm::vec3 &axis) {
    m_position = axis;
    m_mesh->SetPosition(axis);
  }
  void SetVelocity(const glm::vec3 &axis) { m_velocity = axis; }
  void SetAcceleration(const glm::vec3 &axis) { m_acceleration = axis; }
  void SetGravity(const glm::vec3 &gravity) { m_gravity = gravity; }
  void SetMass(float mass) { m_mass = mass; }

 private:
  std::shared_ptr<gfx::Mesh> m_mesh;

  glm::vec3 m_position;
  // glm::vec3 m_orientation;
  glm::vec3 m_velocity;
  // glm::vec3 m_angular_velocity;
  glm::vec3 m_acceleration;
  float m_mass;

  // gravity force
  glm::vec3 m_gravity;
};

template <int num_bodies>
class NbodyScene {
 public:
  // graphics screen
  gfx::Window window;

  // constructs a scene from a window definition
  explicit NbodyScene(gfx::Window t_window) : window(std::move(t_window)) {}

  void OnLoad() {
    crand::SeedRand();

    // initialise m_shader program
    m_shader = std::make_unique<gfx::Shader>("shaders/default.vert",
                                             "shaders/default.frag");
    // initialise m_renderer for this scene
    m_renderer = std::make_unique<gfx::Renderer>();
    m_camera = std::make_unique<gfx::Camera>(glm::vec3(0.0f, 0.0f, 500.0f));

    // initiate the body system
    for (int i = 0; i < num_bodies; ++i) {
      // insantiate a body at random position
      auto body = std::make_shared<Body>();
      body->SetPosition(glm::vec3(crand::GetRand(-200.0f, 200.0f),
                                  crand::GetRand(-100.0f, 100.0f),
                                  crand::GetRand(-50.0f, 50.0f)));
      body->SetMass(1.0f);
      m_bodies.push_back(std::move(body));
    }
  }

  void OnUpdate() {
    using timer::Timer;
    typedef file_io::FileIO fio;

    static std::vector<double> forces_times(MAX_TIME_STEPS);
    static std::vector<double> integration_times(MAX_TIME_STEPS);

    // declare(with default init) a thread pool object
    tpool::std_queue::thread_pool pool{thread_num};
    // Declares a vector of future objects holding the functions that will be
    // executed to perform hash calculation.
    std::vector<std::future<void>> work_items(pool.count());

    std::cout << "\nenter time step #" << gfx::time_step_count << std::endl;
    // compute forces in the n-body system
    {
      static const std::string filename("ComputeForcesTPool.csv");
      // ...
      if (gfx::time_step_count == 0) {
        fio::instance().Save("ComputeForces", filename);
      }
      // ...
      Timer<double, std::milli> forces_timer{};
      // ...
      computeForces(pool, work_items);
      // ...
      const auto exec_time = forces_timer.GetElapsed();
      if (gfx::time_step_count >= 0 && gfx::time_step_count < MAX_TIME_STEPS) {
        forces_times.push_back(exec_time);
      }
      // get the current exuction time
      std::cout << "compute forces (execution time): " << exec_time << " "
                << forces_timer.RatioToString() << std::endl;
      // get the average exuction time
      if (gfx::time_step_count == MAX_TIME_STEPS) {
        const auto avg_exec_time =
            std::accumulate(std::begin(forces_times), std::end(forces_times),
                            0.0) /
            forces_times.size();
        fio::instance().Save(avg_exec_time, filename);
        std::cout << "compute forces ([AVERAGE] execution time)"
                  << avg_exec_time << std::endl;
      }
    }
    // integrate the n-body system
    {
      static const std::string filename("IntegrateBodiesTPool.csv");
      // ...
      if (gfx::time_step_count == 0) {
        fio::instance().Save("IntegrateBodies", filename);
      }
      // ...
      Timer<double, std::milli> integration_timer{};
      // ...
      integrateBodies(pool, work_items);
      const auto exec_time = integration_timer.GetElapsed();
      if (gfx::time_step_count >= 0 && gfx::time_step_count < MAX_TIME_STEPS) {
        integration_times.push_back(exec_time);
      }
      // get the current exuction time
      std::cout << "integrate bodies (execution time): " << exec_time << " "
                << integration_timer.RatioToString() << std::endl;
      // get the average exuction time
      if (gfx::time_step_count == MAX_TIME_STEPS) {
        const auto avg_exec_time =
            std::accumulate(std::begin(integration_times),
                            std::end(integration_times), 0.0) /
            forces_times.size();
        fio::instance().Save(avg_exec_time, filename);
        std::cout << "integrate bodies ([AVERAGE] execution time)"
                  << avg_exec_time << std::endl;
      }
    }

    // temp. for profiling debug
    if (gfx::time_step_count == MAX_TIME_STEPS) {
      std::cout << "Finished profiling\n";
      exit(1);
    }
  }

  void OnDraw() {
    for (int i = 0; i < num_bodies; ++i) {
      m_renderer->Draw(m_shader, m_camera, m_bodies.at(i)->GetMesh());
    }
  }

 private:
  // rendering controllers
  std::unique_ptr<gfx::Camera> m_camera;
  std::unique_ptr<gfx::Shader> m_shader;
  std::unique_ptr<gfx::Renderer> m_renderer;

  // basic physics m_bodies
  std::vector<std::shared_ptr<Body>> m_bodies;

  /* the following functions implement the main n-body compuational parts, and
   * are implemented sequentially, though, in a way that allows for parallel
   * optimisation. */

  // implementation is very gravity-specific at this stage
  void computeForces(tpool::std_queue::thread_pool &pool,
                     std::vector<std::future<void>> &work_items) {
    // Submits task threads in the pool a number of (iterations) times and
    // store.
    std::for_each(
        std::begin(work_items), std::end(work_items), [&](auto &item) {
          item = pool.add_task([this]() {
            // apply gravitational force to each body's velocity
            for (int i = 0; i < num_bodies; ++i) {
              glm::vec3 force_accumulator(0.0f);
              for (int j = 0; j < num_bodies; ++j) {
                // calculate distance between m_bodies
                const glm::vec3 dist = m_bodies.at(j)->GetPosition() -
                                       m_bodies.at(i)->GetPosition();
                const float len = glm::length(dist);
                if (len > 1.0f) {
                  // calculating gravity force's direction
                  const glm::vec3 direction = glm::normalize(dist);
                  // applying Newton's gravity equation: F = G * m1 *
                  // m2 / d^2
                  force_accumulator +=
                      (k_grav *
                       (m_bodies.at(j)->GetMass() * m_bodies.at(i)->GetMass()) /
                       (len * len)) *
                      direction;
                }
              }
              m_bodies.at(i)->SetGravity(force_accumulator);
            }
          });
        });

    // Executes the task threads from the pool.
    std::for_each(std::begin(work_items), std::end(work_items),
                  [](auto &item) { item.get(); });
  }

  // inegrates each body's position
  void integrateBodies(tpool::std_queue::thread_pool &pool,
                       std::vector<std::future<void>> &work_items) {
    // Submits task threads in the pool a number of (iterations) times and
    // store.
    std::for_each(
        std::begin(work_items), std::end(work_items), [&](auto &item) {
          item = pool.add_task([this]() {
            // apply eueler's integration for each body's velocity and
            // acceleration
            // ... maybe angular velocity too
            for (int i = 0; i < num_bodies; ++i) {
              m_bodies.at(i)->SetAcceleration(m_bodies.at(i)->GetGravity());
              m_bodies.at(i)->SetVelocity(
                  m_bodies.at(i)->GetVelocity() +
                  (m_bodies.at(i)->GetAcceleration() * gfx::delta_time));
              m_bodies.at(i)->SetPosition(
                  m_bodies.at(i)->GetPosition() +
                  (m_bodies.at(i)->GetVelocity() * gfx::delta_time));
            }
          });
        });

    // Executes the task threads from the pool.
    std::for_each(std::begin(work_items), std::end(work_items),
                  [](auto &item) { item.get(); });
  }
};

#endif  // NBODY_TPOOL_HPP_

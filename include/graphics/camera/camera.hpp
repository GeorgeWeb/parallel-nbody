#ifndef GRAPHICS_CAMERA_CAMERA_HPP_
#define GRAPHICS_CAMERA_CAMERA_HPP_

#include "../util/glutil.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace graphics {

// default camera values
namespace Defaults {
constexpr float yaw = -85.0f;  // 90 can be problematic
constexpr float pitch = 0.0f;
constexpr float speed = 5.0f;
constexpr float zoom = 45.0f;
};  // namespace Defaults

class Camera {
 public:
  explicit Camera(const glm::vec3 &position = glm::vec3(0.0f)) {
    m_position = position;
    m_front = glm::vec3(0.0f, 0.0f, -1.0f);
    m_world_up = glm::vec3(0.0f, 1.0f, 0.0f);
    updateDirection();
  }

  glm::vec3 GetPosition() { return m_position; }
  void SetPosition(glm::vec3 position) { m_position = position; }

  glm::mat4 GetProjection() {
    static constexpr float aspect_ratio =
        static_cast<float>(1280) / static_cast<float>(720);
    return glm::perspective(glm::radians(Defaults::zoom), aspect_ratio, 0.1f,
                            5000.0f);
  }

  glm::mat4 GetView() {
    return glm::lookAt(m_position, m_position + m_front, m_up);
  }

 private:
  glm::vec3 m_position;
  glm::vec3 m_up;
  glm::vec3 m_front;
  glm::vec3 m_right;
  glm::vec3 m_world_up;

  void updateDirection() {
    // calculate the camera front using the rotational (euler) angles
    m_front = glm::normalize(glm::vec3(
        cos(glm::radians(Defaults::yaw)) * cos(glm::radians(Defaults::pitch)),
        sin(glm::radians(Defaults::pitch)),
        sin(glm::radians(Defaults::yaw)) * cos(glm::radians(Defaults::pitch))));
    m_right = glm::normalize(glm::cross(m_front, m_world_up));
    m_up = glm::normalize(glm::cross(m_right, m_front));
  }
};

}  // namespace graphics

#endif  // GRAPHICS_CAMERA_CAMERA_HPP_

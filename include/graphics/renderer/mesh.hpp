#ifndef GRAPHICS_MESH_HPP_
#define GRAPHICS_MESH_HPP_

#include "../util/glutil.hpp"
#include "index_buffer.hpp"
#include "vertex_array.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <memory>
#include <vector>

namespace graphics {

// custom transform data structure
struct Transform {
  glm::mat4 translate = glm::mat4(1.0f);
  glm::mat4 rotate = glm::mat4(1.0f);
  glm::mat4 scale = glm::mat4(1.0f);

  void Translate(const glm::vec3 &axis) {
    translate = glm::translate(translate, axis);
  }

  void Rotate(float angle, const glm::vec3 &axis) {
    rotate = glm::rotate(rotate, glm::radians(angle), axis);
  }

  void Scale(const glm::vec3 &axis) { scale = glm::scale(scale, axis); }

  glm::mat4 GetModel() const { return translate * rotate * scale; }
};

enum class Shape : int { CUBE = 0 };  // cube's enough at this stage

class Mesh {
 public:
  std::unique_ptr<VertexArray> vao;
  std::unique_ptr<VertexBuffer> vbo;
  std::unique_ptr<IndexBuffer> ibo;
  std::unique_ptr<Transform> transform;

  Mesh(const std::vector<float> &vertices,
       const std::vector<unsigned int> &indices)
      : vao(std::make_unique<VertexArray>()),
        transform(std::make_unique<Transform>()),
        m_vertices(vertices),
        m_indices(indices) {
    setupMesh();
  }

  // generates mesh data based on choice of the provided basic polymesh shapes
  Mesh(Shape shape)
      : vao(std::make_unique<VertexArray>()),
        transform(std::make_unique<Transform>()) {
    switch (shape) {
      case Shape::CUBE: {
        std::vector<glm::vec3> positions;
        // setup vertices
        // front
        positions.push_back(glm::vec3(-1.0f, -1.0f, 1.0f));
        positions.push_back(glm::vec3(1.0f, -1.0f, 1.0f));
        positions.push_back(glm::vec3(1.0f, 1.0f, 1.0f));
        positions.push_back(glm::vec3(-1.0f, 1.0f, 1.0f));
        // back
        positions.push_back(glm::vec3(-1.0f, -1.0f, -1.0f));
        positions.push_back(glm::vec3(1.0f, -1.0f, -1.0f));
        positions.push_back(glm::vec3(1.0f, 1.0f, -1.0f));
        positions.push_back(glm::vec3(-1.0f, 1.0f, -1.0f));

        // setup vertices (in one float data array)
        for (const auto &position : positions) {
          m_vertices.push_back(position.x);
          m_vertices.push_back(position.y);
          m_vertices.push_back(position.z);
        }

        // setup indices
        m_indices = {// front
                     0, 1, 2, 2, 3, 0,
                     // right
                     1, 5, 6, 6, 2, 1,
                     // back
                     7, 6, 5, 5, 4, 7,
                     // left
                     4, 0, 3, 3, 7, 4,
                     // bottom
                     4, 5, 1, 1, 0, 4,
                     // top
                     3, 2, 6, 6, 7, 3};

        setupMesh();
      } break;
      default:
        break;
    }
    transform->Scale(glm::vec3(1.0f));
  }

  std::vector<float> GetVertices() const { return m_vertices; }
  std::vector<unsigned int> GetIndices() const { return m_indices; }

  void Translate(const glm::vec3 &axis) { transform->Translate(axis); }

  void Rotate(float angle, const glm::vec3 &axis) {
    transform->Rotate(angle, axis);
  }

  void Scale(const glm::vec3 &axis) { transform->Scale(axis); }

  /* another way of applying translate transformations by directly changing the
   * values in the matrix to specified 3D position vector: */

  void SetPosition(const glm::vec3 &axis) {
    transform->translate[3].x = axis.x;
    transform->translate[3].y = axis.y;
    transform->translate[3].z = axis.z;
  }

  glm::vec3 GetPosition() const {
    return {transform->translate[3].x, transform->translate[3].y, transform->translate[3].z};
  }

  void Draw() const {
    vao->Bind();
    ibo->Bind();
    constexpr auto mode = GL_TRIANGLES;
    GL_CALL(glDrawElements(mode, ibo->GetCount(), GL_UNSIGNED_INT, nullptr));
  }

 private:
  std::vector<float> m_vertices;
  std::vector<unsigned int> m_indices;

  // generates buffer data and pass it to the rendering pipeline
  inline void setupMesh() {
    vbo.reset(
        new VertexBuffer(m_vertices.data(), m_vertices.size() * sizeof(float)));

    VertexBufferLayout layout;
    layout.Push<float>(3);  // positions
    vao->AddBuffer(*vbo, layout);

    ibo.reset(new IndexBuffer(m_indices.data(), m_indices.size()));
  }
};

}  // namespace graphics

#endif  // GRAPHICS_MESH_HPP_

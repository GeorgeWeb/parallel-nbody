#ifndef NBODY_RENDERER_VERTEX_BUFFER_HPP_
#define NBODY_RENDERER_VERTEX_BUFFER_HPP_

#include "../util/glutil.hpp"

namespace graphics {

class VertexBuffer {
 public:
  VertexBuffer(const void *data, unsigned int size) {
    GL_CALL(glGenBuffers(1, &m_renderer_id));
    GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, m_renderer_id));
    GL_CALL(glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW));
  }

  ~VertexBuffer() { GL_CALL(glDeleteBuffers(1, &m_renderer_id)); }

  void Bind() const { GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, m_renderer_id)); }
  void Unbind() const { GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, 0)); }

 private:
  unsigned int m_renderer_id;
};

}  // namespace graphics

#endif  // NBODY_RENDERER_VERTEX_BUFFER_HPP_
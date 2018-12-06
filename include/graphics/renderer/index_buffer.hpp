#ifndef GRAPHICS_INDEX_BUFFER_HPP_
#define GRAPHICS_INDEX_BUFFER_HPP_

#include "../util/glutil.hpp"

namespace graphics {

class IndexBuffer {
 public:
  IndexBuffer(const unsigned int *data, unsigned int count) {
    ASSERT(sizeof(unsigned int) == sizeof(GLuint));
    GL_CALL(glGenBuffers(1, &m_renderer_id));
    GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_renderer_id));
    GL_CALL(glBufferData(GL_ELEMENT_ARRAY_BUFFER, count * sizeof(unsigned int), data, GL_STATIC_DRAW));
  }

  ~IndexBuffer() { GL_CALL(glDeleteBuffers(1, &m_renderer_id)); }

  void Bind() const { GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_renderer_id)); }
  void Unbind() const { GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)); }

  inline constexpr unsigned int GetCount() const { return m_count; }

 private:
  unsigned int m_renderer_id;
  unsigned int m_count;
};

}  // namespace graphics

#endif  // GRAPHICS_INDEX_BUFFER_HPP_

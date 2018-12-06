#ifndef GRAPHICS_VERTEX_ARRAY_HPP_
#define GRAPHICS_VERTEX_ARRAY_HPP_

#include "../util/glutil.hpp"
#include "vertex_buffer.hpp"
#include "vertex_buffer_layout.hpp"

namespace graphics {

class VertexArray {
 public:
  VertexArray() { GL_CALL(glGenVertexArrays(1, &m_renderer_id)); }
  ~VertexArray() { GL_CALL(glDeleteVertexArrays(1, &m_renderer_id)); }

  void Bind() const { GL_CALL(glBindVertexArray(m_renderer_id)); }
  void Unbind() const { GL_CALL(glBindVertexArray(0)); }

  void AddBuffer(const VertexBuffer &vb,
                 const VertexBufferLayout &layout) const {
    Bind();
    vb.Bind();
    const auto &elements = layout.GetElements();
    int offset = 0;
    for (auto i = 0u; i < elements.size(); ++i) {
      const auto &element = elements[i];
      GL_CALL(glEnableVertexAttribArray(i));
      GL_CALL(glVertexAttribPointer(i, element.count, element.type, element.normalized,
                            layout.GetStride(),
                            reinterpret_cast<const void *>(offset)));
      offset += element.offset;
    }
  }

 private:
  unsigned int m_renderer_id;
};

}  // namespace graphics

#endif  // GRAPHICS_VERTEX_ARRAY_HPP_

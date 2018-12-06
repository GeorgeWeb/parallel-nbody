#ifndef GRAPHICS_VERTEX_BUFFER_LAYOUT_HPP_
#define GRAPHICS_VERTEX_BUFFER_LAYOUT_HPP_

#include "../util/glutil.hpp"

#include <cstdint>
#include <vector>

namespace graphics {

struct VertexBufferElement {
  unsigned int type;
  unsigned int count;
  unsigned int normalized;
  unsigned int offset;

  static unsigned int GetSizeOfType(unsigned int type, unsigned int dim = 1) {
    switch (type) {
      case GL_FLOAT:
        return dim * sizeof(GLfloat);
      case GL_UNSIGNED_INT:
        return dim * sizeof(GLuint);
      case GL_UNSIGNED_BYTE:
        return dim * sizeof(GLubyte);
    }
    ASSERT(false);
    return 0;
  }
};

class VertexBufferLayout {
 public:
  template <typename T>
  void Push(unsigned int) {
    std::cout
        << "This generic function MUST be implemented for desired type.\n";
  }

  inline const std::vector<VertexBufferElement> &GetElements() const {
    return m_elements;
  }
  inline constexpr unsigned int GetStride() const { return m_stride; }

 private:
  std::vector<VertexBufferElement> m_elements;
  unsigned int m_stride{0};
};

template <>
inline void VertexBufferLayout::Push<float>(unsigned int count) {
  const auto offset = count * VertexBufferElement::GetSizeOfType(GL_FLOAT);
  m_stride += offset;
  m_elements.push_back({GL_FLOAT, count, GL_FALSE, offset});
}

template <>
inline void VertexBufferLayout::Push<unsigned int>(unsigned int count) {
  const auto offset =
      count * VertexBufferElement::GetSizeOfType(GL_UNSIGNED_INT);
  m_stride += offset;
  m_elements.push_back({GL_UNSIGNED_INT, count, GL_FALSE, offset});
}

template <>
inline void VertexBufferLayout::Push<unsigned char>(unsigned int count) {
  const auto offset =
      count * VertexBufferElement::GetSizeOfType(GL_UNSIGNED_BYTE);
  m_stride += offset;
  m_elements.push_back({GL_UNSIGNED_BYTE, count, GL_TRUE, offset});
}

}  // namespace graphics

#endif  // GRAPHICS_VERTEX_BUFFER_LAYOUT_HPP_

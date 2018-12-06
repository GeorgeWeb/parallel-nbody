#ifndef GRAPHICS_GLUTIL_HPP_
#define GRAPHICS_GLUTIL_HPP_

#include <glad/glad.h>

// debug break
#include "debugbreak/debugbreak.h"

#include <iostream>

namespace graphics {

#define ASSERT(x) \
  if (!(x)) debug_break();

#define GL_CALL(x) \
  GLClearError();  \
  x;               \
  ASSERT(GLLogCall(#x, __FILE__, __LINE__))

static inline void GLClearError() {
  while (glGetError() != GL_NO_ERROR)
    ;
}

static inline bool GLLogCall(const char *function, const char *file, int line) {
  while (GLenum error = glGetError()) {
    std::cout << "[OpenGL Error]: (" << error << ")" << std::endl;
    std::cout << "\tFunction: " << function << std::endl;
    std::cout << "\tFile: " << file << std::endl;
    std::cout << "\tLine: " << line << std::endl;
    return false;
  }
  return true;
}

}  // namespace graphics

#endif  // GRAPHICS_GLUTIL_HPP_

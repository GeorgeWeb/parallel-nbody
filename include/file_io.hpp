#ifndef FILE_IO_HPP_
#define FILE_IO_HPP_

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>

namespace file_io {

class FileIO {
 public:
  static FileIO &instance() {
    static FileIO instance;
    return instance;
  }

  std::string Read(std::string_view filename) {
    std::stringstream ss;
    try {
      std::ifstream file;
      file.open(std::string(filename), std::ios::in);
      if (!file.fail()) {
        ss << file.rdbuf();
      }
      file.close();
    } catch (const std::exception &e) {
      std::cerr << "ERROR: Cannot open the file: " << filename << " | "
                << e.what() << std::endl;
    }
    return ss.str();
  }

  template <typename T>
  void Save(T data, std::string_view filename) {
    try {
      std::ofstream file;
      file.open(std::string(filename), std::ios::app);
      file.precision(std::numeric_limits<double>::max_digits10);
      file << data << "\n";
      file.close();
    } catch (const std::exception &e) {
      std::cerr << "ERROR: Cannot open the file: " << filename << " | "
                << e.what() << std::endl;
    }
  }
};

}  // namespace file_io

#endif  // FILE_IO_HPP_

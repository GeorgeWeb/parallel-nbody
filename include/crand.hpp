#ifndef CRAND_HPP_
#define CRAND_HPP_

#include <stdlib.h>
#include <time.h>
#include <random>

// c-style random convenience function wrappers
namespace crand {

// abstract in a header file in the root /include dir.
static float GetRand(float lo, float hi) {
  return lo + static_cast<float>(rand()) /
                  (static_cast<float>(RAND_MAX / (hi - lo)));
}

// ...
static void SeedRand(int seed = time(0)) {
  // c-style seeding the rand() generator
  srand(seed);
}

}  // namespace crand

#endif  // CRAND_HPP_

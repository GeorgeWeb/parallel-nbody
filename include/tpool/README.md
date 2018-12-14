# tpool.hpp
A very simple ***single-header*** C++17 implementation of a thread-pool

---

### Compilation
#### Linux / macOS
**gcc**
```
g++ -std=c++17 -pthread example.cpp -o example
```
**clang**
```
clang++ -std=c++17 -pthread example.cpp -o example
```
#### Windows
**Visual Studio 2017** with enabled multi-threading

---

### Usage 1
```cpp
/** This is a 3-step program example that showcases the use of the `tpool` library,
  * and while this is a very simple case, it can be used in a similar way for a
  * in more complex scenarios with multiple iterations over certain tasks
 **/

#include "path/to/tpool.hpp"
#include <iostream>

constexpr float calculate(float value, int scalar) {
  return value * static_cast<float>(scalar);
}

auto main() -> int {
  /* Step 1: Construct a default thread pool object */
  
  // Option 1:
  // implementation with manual (in-place) std::queue::push/pop synchronisation
  // the slighly faster version, which is also considerably safe for most trivial cases
  // code: tpool::std_queue::thread_pool pool{/* number of threads, or leave empty */};
  
  // Option 2:
  // implementation with a thread-safe queue, internally composed of std::queue
  // the slightly safer version, for cases where we need to double-ensue correct ordering
  // code: tpool::safe_queue::thread_pool pool{/* number of threads, or leave empty */};

  // let's go with the first option for this case:
  tpool::std_queue::thread_pool pool{};

  /* Step 2: Set variables to be used in the functions passed to the pool */

  constexpr float num1 = 12.25f;
  constexpr float num2 = 17.75f;
  constexpr int scalar = 2;

  /* Submit tasks to the thread-pool */

  // enqueue lambdas to the pool which will give them a thread if there's any unused threads left
  // there sure are enough, unless you're running on a single-core CPU, where 
  // a default pool construction might create a pool of only 1 thread
  auto foo = pool.enqueue([num1] { return num1 * static_cast<float>(scalar); });
  auto bar = pool.enqueue(calculate, num2, scalar);
  
  /* Step 3: Use the outputs from the tasks that were executed within the thread-pool */

  // the return values from the enqueued lambdas are futures, 
  // so we need to use std::future::get() to retrieve them
  const auto result_foo = foo.get();
  const auto result_bar = bar.get();
  std::cout << "Calculation 1: " << result_foo << std::endl;
  std::cout << "Calculation 2: " << result_bar << std::endl;
  std::cout << "Added together: " << result_foo + result_bar << std::endl;

  return 0;
}
```

**Output**
```
Calculation 1: 24.5
Calculation 2: 35.5
Added together: 60
```

---

### Usage 2
```cpp
/** This is a 3-step program example that showcases the use of the `tpool` library,
  * and while this is a very simple case, it can be used in a similar way for a
  * in more complex scenarios with multiple iterations over certain tasks
 **/

#include "path/to/tpool.hpp"
#include <iostream>
#include <chrono>

auto main () -> int {
  using namespace std;
  using namespace chrono;
  
  constexpr auto iters = 21u;

  auto start = high_resolution_clock::now();

  vector<future<void>> items;
  for(auto i = 0u; i < iters; ++i)
  {
      auto job = pool->enqueue([] {
          this_thread::sleep_for(std::chrono::seconds(1));
      });
      items.push_back(move(job));
  }
  for(auto &item : items) {
      item.get();
  }
  auto end = high_resolution_clock::now();
  duration<double, milli> diff = end - start;
  cout << "Avg task-time: " << diff.count() << " milliseconds\n";

  return 0;
}
```

**Output**
```
TODO: Add program output.
```

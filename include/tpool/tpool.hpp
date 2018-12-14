/** @AUTHOR:
 * Georgi Mirazchiyski
 * ===
 * @LICENSE:
 * NO LICENSE, USE AS YOU WISH!
 * ===
 * @DATE
 * October 17, 2018
 * ===
 * @DESCPRITION:
 * A very simple C++17 implementation of a thread-pool,
 * creates a pool of a specific number of threads, that
 * is useful for certain sitations but lacks optimisations,
 * such as dynamic resizing of the vector of threads, when
 * we may need more of them, or may not need as much as
 * we have allocated initially on construction.
 * ---
 * TODO: This limitation needs tackled in the near future.
 * ---
 * NOTE: It is my first implementation of a thread-pool as well as
 * first exposure to the topic which explains the naive approach here.
 * On the other hand, it simple to read and does the job for trivial cases.
 *
 * Project repository: https://github.com/GeorgeWeb/tpool.hpp
 **/
#ifndef TPOOL_HPP_
#define TPOOL_HPP_

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace tpool {

// forward delcaration of the class interface
template <typename T>
class thread_pool {
 public:
  size_t count() const noexcept { return getUnderlying().count(); }

  template <typename TFunc, typename... TArgs>
  auto add_task(TFunc task, TArgs &&... args)
      -> std::future<decltype(task(args...))> {
    return getUnderlying().template add_task(task, args...);
  }

 protected:
  void start(size_t num_threads) { getUnderlying().start(); }

  void stop() { getUnderlying().stop(); }

 private:
  inline auto &getUnderlying() { return static_cast<T &>(*this); }
};

namespace std_queue {

// implementation using the standard STL queue container for the tasks
class thread_pool : public tpool::thread_pool<tpool::std_queue::thread_pool> {
  using task_t = std::function<void()>;

 public:
  // constuctor with a default number of threads, dependending on your CPU's
  // cores number
  thread_pool() {
    const auto num_threads{
        std::max(std::thread::hardware_concurrency() - 1u, 2u)};
    start(num_threads);
  }

  // constuctor with custom number of threads, dependending on your CPU's cores
  // number
  explicit thread_pool(size_t num_threads) { start(num_threads); }

  ~thread_pool() { stop(); }

  // following the rule of 5, delete the other constructors
  // and their corresponding operator overloads
  thread_pool(const thread_pool &) = delete;
  thread_pool(thread_pool &&) = delete;
  thread_pool &operator=(const thread_pool &) = delete;
  thread_pool &operator=(thread_pool &&) = delete;

  // gets the number of threads in the pool
  size_t count() const noexcept { return _threads.size(); }

  template <typename TFunc, typename... TArgs>
  auto add_task(TFunc task, TArgs &&... args)
      -> std::future<decltype(task(args...))> {
    /** wrapper to the task,
     * where `packaged_task` is the container for the task function/functor,
     * using `shared_ptr`, because the wrapper is shared between tasks inside
     *the thread pool
     **/
    auto func =
        std::bind(std::forward<TFunc>(task), std::forward<TArgs>(args)...);
    auto wrapper =
        std::make_shared<std::packaged_task<decltype(task(args...))()>>(func);

    /** scoped lock
     * safely associate a task to an available thread in the pool
     **/
    {
      std::unique_lock<std::mutex>(_mutex);
      // wrap the packaged task into void function
      _tasks.emplace([wrapper] { (*wrapper)(); });
    }

    // notify an available thread
    _event.notify_one();

    return wrapper->get_future();
  }

 private:
  /* Internal Member Functions */

  void start(size_t num_threads) {
    _threads.reserve(num_threads);
    for (auto i = 0u; i < num_threads; ++i) {
      _threads.emplace_back([=] {
        while (true) {
          task_t task;

          /** scoped lock
           * because we don't know how long a task will run and the mutex
           * may get locked for a long time and block the other threads in the
           *pool, therefore we release the lock on scope exit, then execute the
           *task.
           **/
          {
            std::unique_lock<std::mutex> lock(_mutex);
            // wait until we stop, or if another task has become available
            _event.wait(lock, [this] { return _stopping || !_tasks.empty(); });

            if (_stopping && _tasks.empty()) {
              break;
            }

            task = std::move(_tasks.front());
            _tasks.pop();
          }

          // execute the task
          std::invoke(task);
        }
      });
    }
  }

  void stop() noexcept {
    /** scoped lock
     * safely set the threading mechanism to stop an release the lock on scope
     *exit
     **/
    {
      std::unique_lock<std::mutex> lock(_mutex);
      _stopping = true;
    }

    // stop all waiting threads
    _event.notify_all();

    // finalise the all threads in the pool
    for (auto &thread : _threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  /* Internal Data Members */

  std::vector<std::thread> _threads;
  std::condition_variable _event;

  // used for guarding and synchronisation of threads
  std::mutex _mutex;
  bool _stopping{false};

  // the tasks container
  std::queue<task_t> _tasks;
};

}  // namespace std_queue

namespace safe_queue {

namespace detail {

/** A very basic (and naive) implementation of a thead-safe,
 * lock-based queue container, based on the `std::queue` container
 * The justification of this arguable implementation approach
 * is based on the fact that there won't be a possibility
 * for a deadlock (or livelock with our thread-pool system,
 * because of the nature of its implementation.
 **/
template <typename T>
class queue {
 public:
  void push(T value) {
    std::lock_guard<std::mutex> lock(_mutex);
    _queue.push(value);
    _condition.notify_one();
  }

  // deletes the retrieved element, and returns the front value safely
  std::shared_ptr<T> pop() {
    std::unique_lock<std::mutex> lock(_mutex);
    _condition.wait(lock, [this] { return !_queue.empty(); });
    auto value = std::make_shared<T>(_queue.front());
    _queue.pop();
    return value;
  }

  bool empty() const {
    std::unique_lock<std::mutex> lock(_mutex);
    return _queue.empty();
  }

 private:
  std::queue<T> _queue;
  mutable std::mutex _mutex;
  std::condition_variable _condition;
};

}  // namespace detail

// implementation using a custom thread-safe queue container for the tasks
class thread_pool : public tpool::thread_pool<tpool::safe_queue::thread_pool> {
  using task_t = std::function<void()>;

 public:
  // constuctor with a default number of threads, dependending on your CPU's
  // cores number
  thread_pool() {
    const auto num_threads{
        std::max(std::thread::hardware_concurrency() - 1u, 2u)};
    start(num_threads);
  }

  // constuctor with custom number of threads, dependending on your CPU's cores
  // number
  explicit thread_pool(size_t num_threads) { start(num_threads); }

  ~thread_pool() { stop(); }

  // following the rule of 5, delete the other constructors
  // and their corresponding operator overloads
  thread_pool(const thread_pool &) = delete;
  thread_pool(thread_pool &&) = delete;
  thread_pool &operator=(const thread_pool &) = delete;
  thread_pool &operator=(thread_pool &&) = delete;

  // gets the number of threads in the pool
  size_t count() const noexcept { return _threads.size(); }

  template <typename TFunc, typename... TArgs>
  auto add_task(TFunc task, TArgs &&... args)
      -> std::future<decltype(task(args...))> {
    /** wrapper to the task,
     * where `packaged_task` is the container for the task (std::function) and
     *returns std::future using `shared_ptr`, because the wrapper is shared
     *between tasks inside the thread pool
     **/
    auto func =
        std::bind(std::forward<TFunc>(task), std::forward<TArgs>(args)...);
    auto wrapper =
        std::make_shared<std::packaged_task<decltype(task(args...))()>>(func);

    // add_task the task (the generated parametarised std::function)
    _tasks.push(std::move([wrapper] { (*wrapper)(); }));

    // notify an available thread
    _event.notify_one();

    return wrapper->get_future();
  }

 private:
  /* Internal Member Functions */

  void start(size_t num_threads) {
    _threads.reserve(num_threads);
    for (auto i = 0u; i < num_threads; ++i) {
      _threads.emplace_back([=] {
        while (true) {
          task_t task;

          /** scoped lock
           * because we don't know how long a task will run and the mutex
           * may get locked for a long time and block the other threads in the
           *pool, therefore we release the lock on scope exit, then execute the
           *task.
           **/
          {
            std::unique_lock<std::mutex> lock(_mutex);
            // wait until we stop, or if another task has become available
            _event.wait(lock, [this] { return _stopping || !_tasks.empty(); });

            if (_stopping && _tasks.empty()) {
              break;
            }

            task = std::move(*(_tasks.pop()));
          }

          // execute the task
          std::invoke(task);
        }
      });
    }
  }

  void stop() noexcept {
    /** scoped lock
     * safely set the threading mechanism to stop an release the lock on scope
     *exit
     **/
    {
      std::unique_lock<std::mutex> lock(_mutex);
      _stopping = true;
    }

    // stop all waiting threads
    _event.notify_all();

    // finalise the all threads in the pool
    for (auto &thread : _threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  /* Internal Data Members */

  std::vector<std::thread> _threads;
  std::condition_variable _event;

  // used for guarding and synchronisation of threads
  std::mutex _mutex;
  bool _stopping{false};

  // the tasks container
  detail::queue<task_t> _tasks;
};

}  // namespace safe_queue

}  // namespace tpool

#endif  // TPOOL_HPP_

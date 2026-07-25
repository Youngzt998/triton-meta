//===- SimpleTest.h - minimal header-only C++ test harness ----------------===//
//
// A tiny, dependency-free unit-test harness for the tv validator tests. It
// replaces GoogleTest so the tests build with only the C++ standard library
// (no FetchContent, no network). It offers a small, familiar subset:
//
//   TEST(Suite, Name) { ... }          // define + auto-register a test
//   EXPECT_EQ(a, b)   EXPECT_NE(a, b)
//   EXPECT_TRUE(c)    EXPECT_FALSE(c)
//   ASSERT_TRUE(c)                      // stops the current test on failure
//   EXPECT_THROW(stmt, ExceptionType)
//
// EXPECT_* checks may be followed by `<< "message"` like in GoogleTest. End
// each test file with:  int main() { return simpletest::runAll(); }
//
// A check that fails prints "file:line" and keeps going (EXPECT) or returns
// from the test (ASSERT). runAll() returns 0 only if every check passed.
//
//===----------------------------------------------------------------------===//
#ifndef TV_TEST_VALIDATOR_SIMPLETEST_H
#define TV_TEST_VALIDATOR_SIMPLETEST_H

#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace simpletest {

using TestFn = void (*)();

struct TestCase {
  std::string name;
  TestFn fn;
};

// One registry per test executable (each test file is its own executable).
inline std::vector<TestCase> &registry() {
  static std::vector<TestCase> r;
  return r;
}

// Count of failed checks in the test that is running right now.
inline int &currentFailures() {
  static int n = 0;
  return n;
}

inline int registerTest(std::string name, TestFn fn) {
  registry().push_back({std::move(name), fn});
  return 0;
}

inline void reportFailure(const char *file, int line, const std::string &desc,
                          const std::string &msg) {
  currentFailures()++;
  std::cerr << "    [FAIL] " << file << ":" << line << ": " << desc;
  if (!msg.empty())
    std::cerr << " -- " << msg;
  std::cerr << "\n";
}

// Returned by the EXPECT_* macros. It lets you append `<< "message"` and
// reports the failure once the full expression ends (in the destructor).
class Checker {
public:
  Checker(bool ok, const char *file, int line, std::string desc)
      : failed_(!ok), file_(file), line_(line), desc_(std::move(desc)) {}
  ~Checker() {
    if (failed_)
      reportFailure(file_, line_, desc_, msg_.str());
  }
  template <typename T> Checker &operator<<(const T &v) {
    if (failed_)
      msg_ << v;
    return *this;
  }

private:
  bool failed_;
  const char *file_;
  int line_;
  std::string desc_;
  std::ostringstream msg_;
};

inline int runAll() {
  int passed = 0, failed = 0;
  for (const TestCase &tc : registry()) {
    currentFailures() = 0;
    std::cout << "[ RUN  ] " << tc.name << "\n";
    tc.fn();
    if (currentFailures() == 0) {
      std::cout << "[  OK  ] " << tc.name << "\n";
      passed++;
    } else {
      std::cout << "[ FAIL ] " << tc.name << " (" << currentFailures()
                << " failed checks)\n";
      failed++;
    }
  }
  std::cout << "\n"
            << passed << " passed, " << failed << " failed, "
            << registry().size() << " total\n";
  return failed == 0 ? 0 : 1;
}

} // namespace simpletest

// Define and auto-register a test case.
#define TEST(suite, name)                                                      \
  static void suite##_##name##_body();                                         \
  [[maybe_unused]] static const int suite##_##name##_registrar =               \
      ::simpletest::registerTest(#suite "." #name, &suite##_##name##_body);    \
  static void suite##_##name##_body()

#define EXPECT_TRUE(cond)                                                      \
  ::simpletest::Checker(static_cast<bool>(cond), __FILE__, __LINE__,           \
                        "EXPECT_TRUE(" #cond ")")
#define EXPECT_FALSE(cond)                                                     \
  ::simpletest::Checker(!static_cast<bool>(cond), __FILE__, __LINE__,          \
                        "EXPECT_FALSE(" #cond ")")
#define EXPECT_EQ(a, b)                                                        \
  ::simpletest::Checker((a) == (b), __FILE__, __LINE__,                        \
                        "EXPECT_EQ(" #a ", " #b ")")
#define EXPECT_NE(a, b)                                                        \
  ::simpletest::Checker((a) != (b), __FILE__, __LINE__,                        \
                        "EXPECT_NE(" #a ", " #b ")")

// ASSERT stops the current test (returns) when it fails.
#define ASSERT_TRUE(cond)                                                      \
  do {                                                                         \
    if (!static_cast<bool>(cond)) {                                            \
      ::simpletest::reportFailure(__FILE__, __LINE__,                          \
                                  "ASSERT_TRUE(" #cond ")", "");               \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define EXPECT_THROW(stmt, exType)                                            \
  do {                                                                         \
    bool tvThrew = false;                                                      \
    try {                                                                      \
      stmt;                                                                    \
    } catch (const exType &) {                                                 \
      tvThrew = true;                                                          \
    } catch (...) {                                                            \
    }                                                                          \
    if (!tvThrew)                                                              \
      ::simpletest::reportFailure(                                             \
          __FILE__, __LINE__, "EXPECT_THROW(" #stmt ", " #exType ")", "");     \
  } while (0)

#endif // TV_TEST_VALIDATOR_SIMPLETEST_H

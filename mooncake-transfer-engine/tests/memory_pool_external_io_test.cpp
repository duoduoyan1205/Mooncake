#include <gtest/gtest.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <filesystem>
#include <string>
#include <vector>

#include "memory_pool_transfer_engine.h"

namespace {

namespace fs = std::filesystem;

constexpr int kChildExecFailure = 127;

std::string GetTestDirectory() {
    const char* test_directory = std::getenv("MOONCAKE_MEMORY_POOL_TEST_DIR");
    if (test_directory != nullptr && test_directory[0] != '\0') {
        return test_directory;
    }

    return ".";
}

std::string GetHelperPath(const char* helper_name) {
    const char* helper_directory =
        std::getenv("MOONCAKE_MEMORY_POOL_HELPER_DIR");

    if (helper_directory != nullptr && helper_directory[0] != '\0') {
        return (fs::path(helper_directory) / helper_name).string();
    }

    return (fs::path(GetTestDirectory()) / helper_name).string();
}

bool IsExecutableFile(const std::string& path) {
    return ::access(path.c_str(), X_OK) == 0;
}

class ScopedFd {
public:
    ScopedFd() = default;

    explicit ScopedFd(int fd) : fd_(fd) {}

    ~ScopedFd() {
        Reset();
    }

    ScopedFd(const ScopedFd&) = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;

    ScopedFd(ScopedFd&& other) noexcept : fd_(other.fd_) {
        other.fd_ = -1;
    }

    ScopedFd& operator=(ScopedFd&& other) noexcept {
        if (this != &other) {
            Reset();
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }

    int Get() const {
        return fd_;
    }

    bool Valid() const {
        return fd_ >= 0;
    }

    int Release() {
        int fd = fd_;
        fd_ = -1;
        return fd;
    }

    void Reset(int fd = -1) {
        if (fd_ >= 0) {
            ::close(fd_);
        }
        fd_ = fd;
    }

private:
    int fd_ = -1;
};

class ScopedChildProcess {
public:
    ScopedChildProcess() = default;

    explicit ScopedChildProcess(pid_t pid) : pid_(pid) {}

    ~ScopedChildProcess() {
        if (pid_ > 0) {
            int status = 0;
            while (::waitpid(pid_, &status, 0) < 0 && errno == EINTR) {
            }
        }
    }

    ScopedChildProcess(const ScopedChildProcess&) = delete;
    ScopedChildProcess& operator=(const ScopedChildProcess&) = delete;

    pid_t Get() const {
        return pid_;
    }

    void Reset(pid_t pid = -1) {
        pid_ = pid;
    }

private:
    pid_t pid_ = -1;
};

bool ClearCloseOnExec(int fd) {
    int flags = ::fcntl(fd, F_GETFD);
    if (flags < 0) {
        return false;
    }

    if ((flags & FD_CLOEXEC) == 0) {
        return true;
    }

    return ::fcntl(fd, F_SETFD, flags & ~FD_CLOEXEC) == 0;
}

int RunHelper(const std::string& helper_path,
              const char* operation,
              uint64_t size,
              int dmabuf_fd,
              uint64_t pattern) {
    if (!IsExecutableFile(helper_path)) {
        ADD_FAILURE() << "Helper is not executable: " << helper_path;
        return -1;
    }

    if (!ClearCloseOnExec(dmabuf_fd)) {
        ADD_FAILURE() << "Failed to clear FD_CLOEXEC for DMA-BUF fd "
                      << dmabuf_fd << ": " << std::strerror(errno);
        return -1;
    }

    const std::string size_argument = std::to_string(size);
    const std::string fd_argument = std::to_string(dmabuf_fd);
    const std::string pattern_argument = std::to_string(pattern);

    pid_t pid = ::fork();

    if (pid < 0) {
        ADD_FAILURE() << "fork() failed: " << std::strerror(errno);
        return -1;
    }

    if (pid == 0) {
        ::execl(helper_path.c_str(),
                helper_path.c_str(),
                operation,
                size_argument.c_str(),
                fd_argument.c_str(),
                pattern_argument.c_str(),
                static_cast<char*>(nullptr));

        _exit(kChildExecFailure);
    }

    int status = 0;
    pid_t wait_result;

    do {
        wait_result = ::waitpid(pid, &status, 0);
    } while (wait_result < 0 && errno == EINTR);

    if (wait_result < 0) {
        ADD_FAILURE() << "waitpid() failed: " << std::strerror(errno);
        return -1;
    }

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }

    if (WIFSIGNALED(status)) {
        ADD_FAILURE() << "Helper terminated by signal "
                      << WTERMSIG(status);
        return -1;
    }

    ADD_FAILURE() << "Helper terminated unexpectedly";
    return -1;
}

}  // namespace

TEST(MemoryPoolExternalIoTest, HelperPathsAreAvailable) {
    const std::string gpu_helper =
        GetHelperPath("gpu_dmabuf_helper");

    const std::string nic_helper =
        GetHelperPath("nic_dmabuf_helper");

    GTEST_LOG_(INFO) << "GPU helper path: " << gpu_helper;
    GTEST_LOG_(INFO) << "NIC helper path: " << nic_helper;

    if (!IsExecutableFile(gpu_helper) ||
        !IsExecutableFile(nic_helper)) {
        GTEST_SKIP()
            << "External DMA-BUF helpers are not available. "
            << "Build gpu_dmabuf_helper and nic_dmabuf_helper first.";
    }

    SUCCEED();
}

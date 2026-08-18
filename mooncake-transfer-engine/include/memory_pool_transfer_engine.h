#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace mooncake {

// Thin data-plane adapter for the MPU/SUE Memory Pool device.
// The implementation is intentionally vendor-library agnostic: it loads the
// userspace verbs ABI at runtime and exposes only the operations needed by
// Mooncake Store. Object/replica policy remains in mooncake-store.
class MemoryPoolTransferEngine {
 public:
    struct Allocation {
        uint64_t handle = 0;
        uint64_t global_addr = 0;
        uint64_t size = 0;
    };

    enum class AccessPath : uint32_t {
        kPToD = 0,
        kDToP = 1,
        kDToPool = 2,
        kPoolToD = 3,
    };

    MemoryPoolTransferEngine(std::string sueverbs_library,
                             std::string device_path);
    ~MemoryPoolTransferEngine();

    MemoryPoolTransferEngine(const MemoryPoolTransferEngine&) = delete;
    MemoryPoolTransferEngine& operator=(const MemoryPoolTransferEngine&) = delete;

    int Open();
    void Close();

    bool IsOpen() const { return ctx_ != nullptr; }
    uint64_t Capacity() const { return capacity_; }

    int Allocate(uint64_t size, Allocation* allocation);
    int Free(const Allocation& allocation);

    int Transfer(AccessPath path, uint64_t source_addr, uint64_t target_addr,
                 size_t length);

    int TransferPToD(uint64_t source_addr, uint64_t target_addr, size_t length);
    int TransferDToP(uint64_t source_addr, uint64_t target_addr, size_t length);
    int TransferDToPool(uint64_t source_addr, uint64_t pool_addr, size_t length);
    int TransferPoolToD(uint64_t pool_addr, uint64_t target_addr, size_t length);

    bool LooksLikeDevicePointer(const void* ptr) const;

 private:
    struct Context;
    struct Api;

    int LoadApi();
    void UnloadApi();

    void* library_handle_ = nullptr;
    Context* ctx_ = nullptr;
    std::unique_ptr<Api> api_;
    std::string sueverbs_library_;
    std::string device_path_;
    uint64_t capacity_ = 0;
};

}  // namespace mooncake

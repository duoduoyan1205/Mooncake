#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mooncake {

class MemoryPoolTransferEngine {
 public:
    struct Allocation {
        uint32_t node_id = 0;
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
                             std::string device_paths);
    ~MemoryPoolTransferEngine();

    MemoryPoolTransferEngine(const MemoryPoolTransferEngine&) = delete;
    MemoryPoolTransferEngine& operator=(const MemoryPoolTransferEngine&) = delete;

    // device_paths is a comma-separated list. Each device is treated as one
    // equivalent Memory Pool node. Placement is intentionally round-robin.
    int Open();
    void Close();

    bool IsOpen() const;
    size_t NodeCount() const;
    uint64_t Capacity() const;
    uint64_t NodeCapacity(uint32_t node_id) const;

    int Allocate(uint64_t size, Allocation* allocation);
    int Free(const Allocation& allocation);

    int Transfer(AccessPath path, uint32_t node_id, uint64_t source_addr,
                 uint64_t target_addr, size_t length);

    int TransferPToD(uint32_t node_id, uint64_t source_addr,
                     uint64_t target_addr, size_t length);
    int TransferDToP(uint32_t node_id, uint64_t source_addr,
                     uint64_t target_addr, size_t length);
    int TransferDToPool(const Allocation& allocation, uint64_t source_addr,
                        size_t length, uint64_t offset = 0);
    int TransferPoolToD(const Allocation& allocation, uint64_t target_addr,
                        size_t length, uint64_t offset = 0);

 private:
    struct Context;
    struct Api;
    struct Node;

    int LoadApi();
    void UnloadApi();

    std::unique_ptr<Context> context_;
};

}  // namespace mooncake

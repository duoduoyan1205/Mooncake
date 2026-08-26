#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace mooncake {

class MemoryPoolTransferEngine {
 public:
    struct Allocation {
        uint32_t node_id = 0;
        uint64_t handle = 0;
        uint64_t global_addr = 0;
        uint64_t target_addr = 0;
        uint64_t size = 0;
    };

    MemoryPoolTransferEngine(std::string sueverbs_library,
                             std::string device_paths);
    ~MemoryPoolTransferEngine();

    MemoryPoolTransferEngine(const MemoryPoolTransferEngine&) = delete;
    MemoryPoolTransferEngine& operator=(const MemoryPoolTransferEngine&) = delete;

    // device_paths is a comma-separated list. Each device is one equivalent
    // Memory Pool node. Placement is intentionally round-robin.
    int Open();
    void Close();

    bool IsOpen() const;
    size_t NodeCount() const;
    uint64_t Capacity() const;
    uint64_t NodeCapacity(uint32_t node_id) const;

    int Allocate(uint64_t size, Allocation* allocation);
    int Free(const Allocation& allocation);

    // Return the Memory Pool target address for a range within an allocation.
    int TargetRange(const Allocation& allocation, uint64_t offset,
                    size_t length, uint64_t* target_addr) const;

    // Export the underlying MPU BO as a DMA-BUF. The caller owns the returned
    // file descriptor and must close it when it is no longer needed.
    int ExportDmaBuf(const Allocation& allocation, int flags,
                     int* dmabuf_fd) const;

    // CPU mapping helpers for software-backed MPU pool memory.
    int Map(const Allocation& allocation, size_t offset, size_t length,
            void** cpu_addr);
    int Unmap(const Allocation& allocation, size_t length);

 private:
    struct Api;
    struct Node;
    struct Context;
    std::unique_ptr<Context> context_;
};

}  // namespace mooncake

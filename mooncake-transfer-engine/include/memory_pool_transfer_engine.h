#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <libamdgpu_mpu.h>

namespace mooncake {

class MemoryPoolTransferEngine {
 public:
    // Owns the userspace state associated with one MPU buffer object. The
    // embedded ABI buffer is kept for the allocation lifetime so map/unmap
    // state is never lost between calls.
    struct Allocation {
        uint32_t node_id = 0;
        amdgpu_mpu_buf_t buf{};
        int dmabuf_fd = -1;

        Allocation() = default;
        ~Allocation();
        Allocation(Allocation&& other) noexcept;
        Allocation& operator=(Allocation&& other) noexcept;
        Allocation(const Allocation&) = delete;
        Allocation& operator=(const Allocation&) = delete;

        bool valid() const { return buf.handle != 0; }
        bool mapped() const { return buf.cpu_addr != nullptr; }
    };

    MemoryPoolTransferEngine(std::string sueverbs_library,
                             std::string device_paths);
    ~MemoryPoolTransferEngine();

    MemoryPoolTransferEngine(const MemoryPoolTransferEngine&) = delete;
    MemoryPoolTransferEngine& operator=(const MemoryPoolTransferEngine&) = delete;

    int Open();
    void Close();

    bool IsOpen() const;
    size_t NodeCount() const;
    uint64_t Capacity() const;
    uint64_t NodeCapacity(uint32_t node_id) const;

    int Allocate(uint64_t size, Allocation* allocation);
    int Free(Allocation* allocation);

    int TargetRange(const Allocation& allocation, uint64_t offset,
                    size_t length, uint64_t* target_addr) const;

    // Export the MPU BO as a DMA-BUF. The returned fd is owned by the caller;
    // an internal duplicate is retained by Allocation until Free/destruction.
    int ExportDmaBuf(Allocation* allocation, int flags, int* dmabuf_fd) const;

    // CPU mapping helpers. Mapping state is retained in Allocation::buf so the
    // ABI's mapped_len/cpu_addr checks remain authoritative.
    int Map(Allocation* allocation, size_t offset, size_t length,
            void** cpu_addr);
    int Unmap(Allocation* allocation, size_t length);

 private:
    struct Api;
    struct Node;
    struct Context;
    std::unique_ptr<Context> context_;
};

}  // namespace mooncake

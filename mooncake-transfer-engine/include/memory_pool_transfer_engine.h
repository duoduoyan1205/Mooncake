#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <libamdgpu_mpu.h>

namespace mooncake {

class MemoryPoolTransferEngine {
 public:
    enum class DmaBufType : uint8_t {
        GPU = 0,
        NIC = 1,
    };

    struct ImportedDmaBuf {
        int fd = -1;
        uint64_t length = 0;
        uint64_t address = 0;
        DmaBufType type = DmaBufType::GPU;

        bool valid() const { return fd >= 0 && length != 0; }
    };

    // Allocation is owned by its MemoryPoolTransferEngine. It must be
    // released before the engine is closed or destroyed.
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

    // Export the MPU BO as a DMA-BUF. The returned fd is owned by the caller.
    int ExportDmaBuf(Allocation* allocation, int flags, int* dmabuf_fd) const;

    // Import an externally exported DMA-BUF. The engine duplicates the fd and
    // owns the duplicate until ReleaseDmaBuf(). No CPU mapping is performed.
    // The importer deliberately does not claim that the MPU kernel driver has
    // attached the DMA-BUF: that requires an MPU import ioctl, which is a
    // separate kernel ABI step. This object is the Transfer Engine's lifetime
    // and metadata boundary for the external DMA-BUF.
    int ImportDmaBuf(int dmabuf_fd, uint64_t address, uint64_t length,
                     DmaBufType type, ImportedDmaBuf* imported);
    int ReleaseDmaBuf(ImportedDmaBuf* imported);

    // CPU mapping helpers. Mapping state is retained in Allocation::buf so the
    // ABI's mapped_len/cpu_addr checks remain authoritative.
    int Map(Allocation* allocation, size_t offset, size_t length,
            void** cpu_addr);
    int Unmap(Allocation* allocation, size_t length);

 private:
    struct Node;
    struct Context;
    std::unique_ptr<Context> context_;
};

}  // namespace mooncake

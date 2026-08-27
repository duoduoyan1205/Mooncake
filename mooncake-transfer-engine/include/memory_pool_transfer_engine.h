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

    struct Allocation {
        uint32_t node_id = 0;
        amdgpu_mpu_buf_t buf{};
        int dmabuf_fd = -1;

        Allocation() = default;
        ~Allocation();

        Allocation(Allocation&&) noexcept;
        Allocation& operator=(Allocation&&) noexcept;

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

    int TargetRange(const Allocation& allocation,
                    uint64_t offset,
                    size_t length,
                    uint64_t* target_address) const;

    int ExportDmaBuf(Allocation* allocation,
                     int flags,
                     int* dma_buf_fd) const;

    int ImportDmaBuf(int fd,
                     uint64_t address,
                     uint64_t length,
                     DmaBufType type,
                     ImportedDmaBuf* imported) const;

    int ReleaseDmaBuf(ImportedDmaBuf* imported) const;

    int SubmitDmaBufTransfer(int fd,
                             uint64_t target_address,
                             uint64_t source_offset,
                             uint64_t length,
                             uint32_t op,
                             uint64_t* cookie);

    int GetDmaBufTransferStatus(uint64_t cookie,
                                amdgpu_mpu_sue_status_t* status);

    int Map(Allocation* allocation,
            size_t offset,
            size_t length,
            void** address);

    int Unmap(Allocation* allocation, size_t length);

private:
    struct Node;
    struct Context;

    std::unique_ptr<Context> context_;
};

}  // namespace mooncake

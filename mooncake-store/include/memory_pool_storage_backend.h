#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "storage_backend.h"

namespace mooncake {

// Mooncake Store backend for the AMDGPU MPU Memory Pool Node.
//
// The current MPU DRM UAPI owns node-memory allocations through GEM/DRM and
// returns both an MPU-local GPU VA and an initiator-visible target address.
// Data movement is intentionally not faked in this backend: the old private
// XFER ioctl was removed from the driver. A subsequent transport integration
// will consume the target address through the GPU/NIC DMA-BUF/SUE path.
class MemoryPoolStorageBackend final : public StorageBackendInterface {
   public:
    explicit MemoryPoolStorageBackend(const FileStorageConfig& config);
    ~MemoryPoolStorageBackend() override;

    tl::expected<void, ErrorCode> Init() override;

    tl::expected<int64_t, ErrorCode> BatchOffload(
        const std::unordered_map<std::string, std::vector<Slice>>& batch_object,
        std::function<ErrorCode(const std::vector<std::string>& keys,
                                std::vector<StorageObjectMetadata>& metadatas)>
            complete_handler,
        EvictionHandler eviction_handler = nullptr) override;

    tl::expected<void, ErrorCode> BatchLoad(
        std::unordered_map<std::string, Slice>& batched_slices) override;

    tl::expected<bool, ErrorCode> IsExist(const std::string& key) override;
    tl::expected<bool, ErrorCode> IsEnableOffloading() override;

    tl::expected<void, ErrorCode> ScanMeta(
        const std::function<ErrorCode(
            const std::vector<std::string>& keys,
            std::vector<StorageObjectMetadata>& metadatas)>& handler) override;

    void RemoveAll() override;

    tl::expected<std::vector<std::string>, ErrorCode> EvictAboveDiskWatermark(
        double high_watermark_ratio, double low_watermark_ratio,
        EvictionHandler eviction_handler = nullptr) override;

    bool IsMemoryPoolBackend() const { return true; }

   private:
    struct Allocation {
        uint32_t handle = 0;
        uint64_t target_addr = 0;
        uint64_t gpu_addr = 0;
        uint64_t size = 0;
        uint64_t mmap_offset = 0;
    };

    struct Entry {
        Allocation allocation;
        uint64_t sequence = 0;
    };

    tl::expected<void, ErrorCode> OpenDevice();
    void CloseDevice();

    tl::expected<Allocation, ErrorCode> Allocate(uint64_t size);
    tl::expected<void, ErrorCode> Free(const Allocation& allocation);

    tl::expected<void, ErrorCode> TransferDeviceToPool(
        const Slice& source, const Allocation& destination, uint64_t offset);
    tl::expected<void, ErrorCode> TransferPoolToDevice(
        const Allocation& source, const Slice& destination, uint64_t offset);

    tl::expected<void, ErrorCode> EvictForSpace(
        uint64_t required_size, EvictionHandler eviction_handler);

    int fd_ = -1;
    std::string device_path_;
    uint64_t capacity_ = 0;
    std::atomic<uint64_t> used_bytes_{0};
    std::atomic<uint64_t> next_sequence_{0};
    std::atomic<bool> initialized_{false};

    mutable std::mutex mutex_;
    std::unordered_map<std::string, Entry> entries_;
};

}  // namespace mooncake

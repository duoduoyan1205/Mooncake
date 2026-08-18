#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "storage_backend.h"

namespace mooncake {

class MemoryPoolStorageBackend final : public StorageBackendInterface {
 public:
    explicit MemoryPoolStorageBackend(const FileStorageConfig& config);
    ~MemoryPoolStorageBackend() override;

    tl::expected<void, ErrorCode> Init() override;
    tl::expected<int64_t, ErrorCode> BatchOffload(
        const std::unordered_map<std::string, std::vector<Slice>>& batch_object,
        std::function<ErrorCode(const std::vector<std::string>&,
                                std::vector<StorageObjectMetadata>&)> complete_handler,
        EvictionHandler eviction_handler = nullptr) override;
    tl::expected<void, ErrorCode> BatchLoad(
        std::unordered_map<std::string, Slice>& batched_slices) override;
    tl::expected<bool, ErrorCode> IsExist(const std::string& key) override;
    tl::expected<bool, ErrorCode> IsEnableOffloading() override;
    tl::expected<void, ErrorCode> ScanMeta(
        const std::function<ErrorCode(const std::vector<std::string>&,
                                      std::vector<StorageObjectMetadata>&)>& handler) override;
    void RemoveAll() override;
    tl::expected<std::vector<std::string>, ErrorCode> EvictAboveDiskWatermark(
        double high_watermark_ratio, double low_watermark_ratio,
        EvictionHandler eviction_handler = nullptr) override;

 private:
    struct Allocation { uint64_t handle = 0; uint64_t global_addr = 0; uint64_t size = 0; };
    struct Entry { Allocation allocation; uint64_t sequence = 0; };

    tl::expected<void, ErrorCode> OpenDevice();
    void CloseDevice();
    tl::expected<Allocation, ErrorCode> Allocate(uint64_t size);
    tl::expected<void, ErrorCode> Free(const Allocation& allocation);
    tl::expected<void, ErrorCode> Transfer(const Slice& slice, const Allocation& allocation,
                                           uint64_t offset, bool to_pool);
    tl::expected<void*, ErrorCode> MapAllocation(const Allocation& allocation);
    void UnmapAllocation(void* address, uint64_t size);
    bool LooksLikeDevicePointer(const void* ptr) const;
    tl::expected<void, ErrorCode> EvictForSpace(uint64_t required_size,
                                                EvictionHandler eviction_handler);

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

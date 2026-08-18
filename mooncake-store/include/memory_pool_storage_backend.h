#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "memory_pool_transfer_engine.h"
#include "storage_backend.h"

namespace mooncake {

enum class MemoryPoolAccessPath : uint32_t {
    kPToD = 0,
    kDToP = 1,
    kDToPool = 2,
    kPoolToD = 3,
};

// Memory Pool is a volatile remote-memory backend.  It reuses the Store's
// object/key interface, but its physical lifecycle is memory allocation and
// DMA transfer, not file creation, eviction, or filesystem persistence.
class MemoryPoolStorageBackend final : public StorageBackendInterface {
 public:
    using Allocation = MemoryPoolTransferEngine::Allocation;

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

    // Direct transfer-engine entry points used by the four MPU access paths.
    tl::expected<void, ErrorCode> TransferPToD(const Slice& src, const Slice& dst);
    tl::expected<void, ErrorCode> TransferDToP(const Slice& src, const Slice& dst);
    tl::expected<void, ErrorCode> TransferDToPool(
        const Slice& src, const Allocation& allocation, uint64_t offset = 0);
    tl::expected<void, ErrorCode> TransferPoolToD(
        const Slice& dst, const Allocation& allocation, uint64_t offset = 0);

    // CPU-side allocation metadata lookup.  The Memory Pool itself stores
    // only KV data; key/hash and allocation descriptors remain in Store DRAM.
    tl::expected<Allocation, ErrorCode> GetAllocation(
        const std::string& key) const;

 private:
    struct Entry {
        Allocation allocation;
    };

    tl::expected<Allocation, ErrorCode> Allocate(uint64_t size);
    tl::expected<void, ErrorCode> Free(const Allocation& allocation);
    tl::expected<void, ErrorCode> Transfer(const Slice& slice,
                                            const Allocation& allocation,
                                            uint64_t offset, bool to_pool);
    tl::expected<void, ErrorCode> TransferGpuToGpu(const Slice& src,
                                                    const Slice& dst,
                                                    MemoryPoolAccessPath path);
    bool LooksLikeDevicePointer(const void* ptr) const;

    std::unique_ptr<MemoryPoolTransferEngine> transfer_engine_;
    uint64_t capacity_ = 0;
    std::atomic<uint64_t> used_bytes_{0};
    std::atomic<bool> initialized_{false};
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Entry> entries_;
};

}  // namespace mooncake

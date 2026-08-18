#include "memory_pool_storage_backend.h"

#include <algorithm>
#include <cstdlib>
#include <string>

#include "device/accelerator_registry.h"

namespace mooncake {
namespace {
std::string DevicePath() {
    const char* p = std::getenv("MOONCAKE_MEMORY_POOL_DEVICES");
    if (p && *p) return p;
    p = std::getenv("MOONCAKE_MEMORY_POOL_DEVICE");
    return p && *p ? p : "/dev/amdgpu-mpu";
}

std::string SueVerbsLibrary() {
    const char* p = std::getenv("MOONCAKE_SUEVERBS_LIBRARY");
    return p && *p ? p : "libsueverbs.so";
}
}  // namespace

MemoryPoolStorageBackend::MemoryPoolStorageBackend(
    const FileStorageConfig& config)
    : StorageBackendInterface(config),
      transfer_engine_(std::make_unique<MemoryPoolTransferEngine>(
          SueVerbsLibrary(), DevicePath())),
      capacity_(static_cast<uint64_t>(
          std::max<int64_t>(0, config.total_size_limit))) {}

MemoryPoolStorageBackend::~MemoryPoolStorageBackend() { RemoveAll(); }

tl::expected<MemoryPoolStorageBackend::Allocation, ErrorCode>
MemoryPoolStorageBackend::Allocate(uint64_t size) {
    if (!transfer_engine_ || !transfer_engine_->IsOpen() || !size) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    Allocation allocation;
    if (transfer_engine_->Allocate(size, &allocation) != 0) {
        return tl::make_unexpected(ErrorCode::BUFFER_OVERFLOW);
    }
    used_bytes_.fetch_add(allocation.size, std::memory_order_relaxed);
    return allocation;
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::Free(
    const Allocation& allocation) {
    if (!transfer_engine_ || !allocation.handle) return {};
    if (transfer_engine_->Free(allocation) != 0) {
        return tl::make_unexpected(ErrorCode::TRANSFER_FAIL);
    }
    used_bytes_.fetch_sub(allocation.size, std::memory_order_relaxed);
    return {};
}

bool MemoryPoolStorageBackend::LooksLikeDevicePointer(const void* ptr) const {
    if (!ptr) return false;
    auto& registry = device::GetAcceleratorRegistry().RuntimeAccelerators();
    device::PointerInfo info{};
    return registry.FindDeviceForPointer(const_cast<void*>(ptr), &info) != nullptr;
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::TransferGpuToGpu(
    const Slice& src, const Slice& dst, MemoryPoolAccessPath path) {
    if (!transfer_engine_ || !transfer_engine_->IsOpen() ||
        !LooksLikeDevicePointer(src.ptr) || !LooksLikeDevicePointer(dst.ptr) ||
        !src.size || !dst.size || src.size != dst.size) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    int rc = -1;
    switch (path) {
        case MemoryPoolAccessPath::kPToD:
            rc = transfer_engine_->TransferPToD(
                0, reinterpret_cast<uint64_t>(src.ptr),
                reinterpret_cast<uint64_t>(dst.ptr), src.size);
            break;
        case MemoryPoolAccessPath::kDToP:
            rc = transfer_engine_->TransferDToP(
                0, reinterpret_cast<uint64_t>(src.ptr),
                reinterpret_cast<uint64_t>(dst.ptr), src.size);
            break;
        default:
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    if (rc != 0) return tl::make_unexpected(ErrorCode::TRANSFER_FAIL);
    return {};
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::TransferPToD(
    const Slice& src, const Slice& dst) {
    return TransferGpuToGpu(src, dst, MemoryPoolAccessPath::kPToD);
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::TransferDToP(
    const Slice& src, const Slice& dst) {
    return TransferGpuToGpu(src, dst, MemoryPoolAccessPath::kDToP);
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::Transfer(
    const Slice& slice, const Allocation& allocation, uint64_t offset,
    bool to_pool) {
    if (!transfer_engine_ || !transfer_engine_->IsOpen() ||
        !LooksLikeDevicePointer(slice.ptr) || !slice.size ||
        offset > allocation.size || slice.size > allocation.size - offset) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    const int rc = to_pool
                       ? transfer_engine_->TransferDToPool(
                             allocation, reinterpret_cast<uint64_t>(slice.ptr),
                             slice.size, offset)
                       : transfer_engine_->TransferPoolToD(
                             allocation, reinterpret_cast<uint64_t>(slice.ptr),
                             slice.size, offset);
    if (rc != 0) return tl::make_unexpected(ErrorCode::TRANSFER_FAIL);
    return {};
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::TransferDToPool(
    const Slice& src, const Allocation& allocation, uint64_t offset) {
    return Transfer(src, allocation, offset, true);
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::TransferPoolToD(
    const Slice& dst, const Allocation& allocation, uint64_t offset) {
    return Transfer(dst, allocation, offset, false);
}

tl::expected<MemoryPoolStorageBackend::Allocation, ErrorCode>
MemoryPoolStorageBackend::GetAllocation(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return tl::make_unexpected(ErrorCode::OBJECT_NOT_FOUND);
    }
    return it->second.allocation;
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::Init() {
    bool expected = false;
    if (!initialized_.compare_exchange_strong(expected, true)) return {};
    if (!transfer_engine_) {
        initialized_.store(false, std::memory_order_release);
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }

    const int rc = transfer_engine_->Open();
    if (rc != 0) {
        initialized_.store(false, std::memory_order_release);
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }

    capacity_ = capacity_ ? std::min(capacity_, transfer_engine_->Capacity())
                          : transfer_engine_->Capacity();
    if (!capacity_) {
        transfer_engine_->Close();
        initialized_.store(false, std::memory_order_release);
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    return {};
}

tl::expected<int64_t, ErrorCode> MemoryPoolStorageBackend::BatchOffload(
    const std::unordered_map<std::string, std::vector<Slice>>& objects,
    std::function<ErrorCode(const std::vector<std::string>&,
                            std::vector<StorageObjectMetadata>&)> complete,
    EvictionHandler /*handler*/) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }

    // Memory Pool has no backend-owned eviction policy.  Space pressure is
    // reported as allocation failure; KV eviction remains a Store/scheduler
    // decision.
    for (const auto& [key, slices] : objects) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (entries_.find(key) != entries_.end()) {
            return tl::make_unexpected(ErrorCode::OBJECT_ALREADY_EXISTS);
        }
        (void)slices;
    }

    std::vector<std::string> keys;
    std::vector<StorageObjectMetadata> metadata;
    std::vector<std::pair<std::string, Allocation>> created;

    for (const auto& [key, slices] : objects) {
        uint64_t total = 0;
        for (const auto& slice : slices) total += slice.size;
        if (!total) continue;

        auto allocation = Allocate(total);
        if (!allocation) {
            for (const auto& [created_key, created_allocation] : created) {
                (void)Free(created_allocation);
                std::lock_guard<std::mutex> lock(mutex_);
                entries_.erase(created_key);
            }
            return tl::make_unexpected(allocation.error());
        }

        uint64_t offset = 0;
        bool ok = true;
        for (const auto& slice : slices) {
            auto transfer = TransferDToPool(slice, allocation.value(), offset);
            if (!transfer) {
                ok = false;
                break;
            }
            offset += slice.size;
        }
        if (!ok) {
            (void)Free(allocation.value());
            for (const auto& [created_key, created_allocation] : created) {
                (void)Free(created_allocation);
                std::lock_guard<std::mutex> lock(mutex_);
                entries_.erase(created_key);
            }
            return tl::make_unexpected(ErrorCode::TRANSFER_FAIL);
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            entries_.emplace(key, Entry{allocation.value()});
        }
        created.emplace_back(key, allocation.value());
        keys.push_back(key);

        // StorageObjectMetadata is a Store-wide compatibility structure.  The
        // authoritative Memory Pool descriptor remains in entries_ in CPU DRAM;
        // no filesystem or persistent-storage metadata is created here.
        metadata.push_back(StorageObjectMetadata{
            static_cast<int64_t>(allocation.value().node_id),
            static_cast<int64_t>(allocation.value().global_addr),
            static_cast<int64_t>(allocation.value().handle),
            static_cast<int64_t>(total),
            ""});
    }

    if (complete && !keys.empty()) {
        auto result = complete(keys, metadata);
        if (result != ErrorCode::OK) {
            for (const auto& [created_key, created_allocation] : created) {
                (void)Free(created_allocation);
                std::lock_guard<std::mutex> lock(mutex_);
                entries_.erase(created_key);
            }
            return tl::make_unexpected(result);
        }
    }
    return static_cast<int64_t>(keys.size());
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::BatchLoad(
    std::unordered_map<std::string, Slice>& objects) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }

    for (auto& [key, dst] : objects) {
        Allocation allocation;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = entries_.find(key);
            if (it == entries_.end()) {
                return tl::make_unexpected(ErrorCode::OBJECT_NOT_FOUND);
            }
            allocation = it->second.allocation;
        }
        if (dst.size > allocation.size) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        auto result = TransferPoolToD(dst, allocation, 0);
        if (!result) return result;
    }
    return {};
}

tl::expected<bool, ErrorCode> MemoryPoolStorageBackend::IsExist(
    const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.find(key) != entries_.end();
}

tl::expected<bool, ErrorCode> MemoryPoolStorageBackend::IsEnableOffloading() {
    return initialized_.load(std::memory_order_acquire);
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::ScanMeta(
    const std::function<ErrorCode(const std::vector<std::string>&,
                                  std::vector<StorageObjectMetadata>&)>& handler) {
    std::vector<std::string> keys;
    std::vector<StorageObjectMetadata> metadata;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        keys.reserve(entries_.size());
        metadata.reserve(entries_.size());
        for (const auto& [key, entry] : entries_) {
            keys.push_back(key);
            metadata.push_back(StorageObjectMetadata{
                static_cast<int64_t>(entry.allocation.node_id),
                static_cast<int64_t>(entry.allocation.global_addr),
                static_cast<int64_t>(entry.allocation.handle),
                static_cast<int64_t>(entry.allocation.size),
                ""});
        }
    }
    if (handler && !keys.empty()) {
        auto result = handler(keys, metadata);
        if (result != ErrorCode::OK) return tl::make_unexpected(result);
    }
    return {};
}

void MemoryPoolStorageBackend::RemoveAll() {
    std::vector<Allocation> allocations;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        allocations.reserve(entries_.size());
        for (const auto& [key, entry] : entries_) {
            (void)key;
            allocations.push_back(entry.allocation);
        }
        entries_.clear();
    }
    for (const auto& allocation : allocations) (void)Free(allocation);
}

}  // namespace mooncake

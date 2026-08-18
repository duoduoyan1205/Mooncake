#include "memory_pool_storage_backend.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "device/accelerator_registry.h"

namespace mooncake {
namespace {
constexpr uint64_t kAlignment = 4096;
constexpr uint64_t kTimeoutNs = 30ULL * 1000 * 1000 * 1000;

uint64_t AlignUp(uint64_t v) {
    return (v + kAlignment - 1) & ~(kAlignment - 1);
}

std::string DevicePath() {
    const char* p = std::getenv("MOONCAKE_MEMORY_POOL_DEVICE");
    return p && *p ? p : "/dev/amdgpu-mpu";
}
}  // namespace

MemoryPoolStorageBackend::MemoryPoolStorageBackend(
    const FileStorageConfig& config)
    : StorageBackendInterface(config),
      device_path_(DevicePath()),
      capacity_(static_cast<uint64_t>(
          std::max<int64_t>(0, config.total_size_limit))) {}

MemoryPoolStorageBackend::~MemoryPoolStorageBackend() {
    RemoveAll();
    CloseDevice();
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::OpenDevice() {
    if (fd_ >= 0) return {};

    fd_ = open(device_path_.c_str(), O_RDWR | O_CLOEXEC);
    if (fd_ < 0) {
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }

    amdgpu_mpu::IoctlCaps caps{};
    if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_CAPS, &caps) == 0 &&
        caps.mem_size) {
        capacity_ = capacity_ ? std::min(capacity_, caps.mem_size)
                              : caps.mem_size;
    }
    if (!capacity_) {
        close(fd_);
        fd_ = -1;
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    return {};
}

void MemoryPoolStorageBackend::CloseDevice() {
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

tl::expected<MemoryPoolStorageBackend::Allocation, ErrorCode>
MemoryPoolStorageBackend::Allocate(uint64_t size) {
    amdgpu_mpu::IoctlAlloc req{AlignUp(size), kAlignment, 0, 0};
    if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_ALLOC, &req) < 0) {
        return tl::make_unexpected(ErrorCode::BUFFER_OVERFLOW);
    }
    return Allocation{req.handle, req.global_addr, req.size};
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::Free(
    const Allocation& allocation) {
    if (!allocation.handle) return {};
    amdgpu_mpu::IoctlFree req{allocation.handle};
    if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_FREE, &req) < 0) {
        return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    }
    return {};
}

bool MemoryPoolStorageBackend::LooksLikeDevicePointer(const void* ptr) const {
    if (!ptr) return false;
    auto& registry = device::GetAcceleratorRegistry().RuntimeAccelerators();
    device::PointerInfo info{};
    return registry.FindDeviceForPointer(const_cast<void*>(ptr), &info) != nullptr;
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::TransferGpuToGpu(
    const Slice& src, const Slice& dst, uint32_t path) {
    if (!LooksLikeDevicePointer(src.ptr) || !LooksLikeDevicePointer(dst.ptr) ||
        !src.size || !dst.size || src.size != dst.size) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    amdgpu_mpu::IoctlXfer req{};
    req.src_addr = reinterpret_cast<uint64_t>(src.ptr);
    req.dst_addr = reinterpret_cast<uint64_t>(dst.ptr);
    req.length = src.size;
    req.path = path;
    req.flags = amdgpu_mpu::kXferSignal | amdgpu_mpu::kXferOrdered;
    req.fence_fd = -1;
    if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_XFER, &req) < 0) {
        return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    }
    if (req.cookie) {
        amdgpu_mpu::IoctlWait wait{req.cookie, kTimeoutNs, 0, 0};
        if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_WAIT, &wait) < 0 ||
            wait.status) {
            return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
        }
    }
    return {};
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::TransferPToD(
    const Slice& src, const Slice& dst) {
    return TransferGpuToGpu(src, dst, amdgpu_mpu::kPathPToD);
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::TransferDToP(
    const Slice& src, const Slice& dst) {
    return TransferGpuToGpu(src, dst, amdgpu_mpu::kPathDToP);
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::Transfer(
    const Slice& slice, const Allocation& allocation, uint64_t offset,
    bool to_pool) {
    if (!LooksLikeDevicePointer(slice.ptr) || !slice.size ||
        offset > allocation.size || slice.size > allocation.size - offset) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    amdgpu_mpu::IoctlXfer req{};
    req.src_addr = to_pool
                       ? reinterpret_cast<uint64_t>(slice.ptr)
                       : allocation.global_addr + offset;
    req.dst_addr = to_pool
                       ? allocation.global_addr + offset
                       : reinterpret_cast<uint64_t>(slice.ptr);
    req.length = slice.size;
    req.path = to_pool ? amdgpu_mpu::kPathDToPoolWrite
                       : amdgpu_mpu::kPathDToPoolRead;
    req.flags = amdgpu_mpu::kXferSignal | amdgpu_mpu::kXferOrdered;
    req.fence_fd = -1;
    if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_XFER, &req) < 0) {
        return tl::make_unexpected(to_pool ? ErrorCode::FILE_WRITE_FAIL
                                           : ErrorCode::FILE_READ_FAIL);
    }
    if (req.cookie) {
        amdgpu_mpu::IoctlWait wait{req.cookie, kTimeoutNs, 0, 0};
        if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_WAIT, &wait) < 0 ||
            wait.status) {
            return tl::make_unexpected(to_pool ? ErrorCode::FILE_WRITE_FAIL
                                               : ErrorCode::FILE_READ_FAIL);
        }
    }
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

tl::expected<void, ErrorCode>
MemoryPoolStorageBackend::RegisterRemoteAllocation(
    const std::string& key, const Allocation& allocation) {
    if (key.empty() || !allocation.handle || !allocation.global_addr ||
        !allocation.size) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    if (!initialized_.load(std::memory_order_acquire)) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(key);
    if (it != entries_.end() && it->second.local_owner) {
        used_bytes_.fetch_sub(it->second.allocation.size,
                              std::memory_order_relaxed);
    }
    entries_[key] = Entry{allocation, next_sequence_.fetch_add(1), false};
    return {};
}

tl::expected<void, ErrorCode>
MemoryPoolStorageBackend::UnregisterRemoteAllocation(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return tl::make_unexpected(ErrorCode::OBJECT_NOT_FOUND);
    }
    if (it->second.local_owner) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    entries_.erase(it);
    return {};
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

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::EvictForSpace(
    uint64_t required, EvictionHandler handler) {
    while (used_bytes_.load(std::memory_order_relaxed) + required > capacity_) {
        Entry victim;
        std::string key;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = std::find_if(entries_.begin(), entries_.end(),
                                   [](const auto& item) {
                                       return item.second.local_owner;
                                   });
            if (it == entries_.end()) {
                return tl::make_unexpected(ErrorCode::BUFFER_OVERFLOW);
            }
            for (auto candidate = entries_.begin(); candidate != entries_.end();
                 ++candidate) {
                if (candidate->second.local_owner &&
                    candidate->second.sequence < it->second.sequence) {
                    it = candidate;
                }
            }
            key = it->first;
            victim = it->second;
        }
        if (handler) {
            auto r = handler({key});
            if (!r) return tl::make_unexpected(r.error());
        }
        auto r = Free(victim.allocation);
        if (!r) return r;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(key);
        if (it != entries_.end() && it->second.sequence == victim.sequence) {
            used_bytes_.fetch_sub(it->second.allocation.size,
                                  std::memory_order_relaxed);
            entries_.erase(it);
        }
    }
    return {};
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::Init() {
    bool expected = false;
    if (!initialized_.compare_exchange_strong(expected, true)) return {};
    auto result = OpenDevice();
    if (!result) initialized_.store(false, std::memory_order_release);
    return result;
}

tl::expected<int64_t, ErrorCode> MemoryPoolStorageBackend::BatchOffload(
    const std::unordered_map<std::string, std::vector<Slice>>& objects,
    std::function<ErrorCode(
        const std::vector<std::string>&,
        std::vector<StorageObjectMetadata>&)> complete,
    EvictionHandler handler) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }

    uint64_t bytes = 0;
    for (const auto& [key, slices] : objects) {
        (void)key;
        for (const auto& slice : slices) bytes += slice.size;
    }
    auto eviction_result = EvictForSpace(bytes, handler);
    if (!eviction_result) {
        return tl::make_unexpected(eviction_result.error());
    }

    std::vector<std::string> keys;
    std::vector<StorageObjectMetadata> metadata;
    for (const auto& [key, slices] : objects) {
        uint64_t total = 0;
        for (const auto& slice : slices) total += slice.size;
        if (!total) continue;

        auto allocation = Allocate(total);
        if (!allocation) continue;

        uint64_t offset = 0;
        bool ok = true;
        for (const auto& slice : slices) {
            auto transfer = TransferDToPool(
                slice, allocation.value(), offset);
            if (!transfer) {
                ok = false;
                break;
            }
            offset += slice.size;
        }
        if (!ok) {
            (void)Free(allocation.value());
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            entries_[key] = Entry{allocation.value(),
                                  next_sequence_.fetch_add(1), true};
        }
        used_bytes_.fetch_add(allocation.value().size,
                              std::memory_order_relaxed);
        keys.push_back(key);
        // StorageObjectMetadata has no dedicated Memory Pool fields. Keep the
        // allocation handle in bucket_id, the global address in offset, and
        // the object size in data_size. The replica descriptor is the canonical
        // strongly-typed representation used by MasterClient.
        metadata.push_back(StorageObjectMetadata{
            static_cast<int64_t>(allocation.value().handle),
            static_cast<int64_t>(allocation.value().global_addr), 0,
            static_cast<int64_t>(total), ""});
    }

    if (complete && !keys.empty()) {
        auto result = complete(keys, metadata);
        if (result != ErrorCode::OK) {
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
        Entry entry;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = entries_.find(key);
            if (it == entries_.end()) {
                return tl::make_unexpected(ErrorCode::OBJECT_NOT_FOUND);
            }
            entry = it->second;
        }
        if (dst.size > entry.allocation.size) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        auto result = TransferPoolToD(dst, entry.allocation, 0);
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
    const std::function<ErrorCode(
        const std::vector<std::string>&,
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
                static_cast<int64_t>(entry.allocation.handle),
                static_cast<int64_t>(entry.allocation.global_addr), 0,
                static_cast<int64_t>(entry.allocation.size), ""});
        }
    }
    if (handler && !keys.empty()) {
        auto result = handler(keys, metadata);
        if (result != ErrorCode::OK) {
            return tl::make_unexpected(result);
        }
    }
    return {};
}

void MemoryPoolStorageBackend::RemoveAll() {
    std::vector<Allocation> local_allocations;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [key, entry] : entries_) {
            if (entry.local_owner) local_allocations.push_back(entry.allocation);
        }
        entries_.clear();
        used_bytes_.store(0, std::memory_order_relaxed);
    }
    for (const auto& allocation : local_allocations) {
        (void)Free(allocation);
    }
}

tl::expected<std::vector<std::string>, ErrorCode>
MemoryPoolStorageBackend::EvictAboveDiskWatermark(
    double high, double low, EvictionHandler handler) {
    if (high <= 0 || low < 0 || low >= high) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    const uint64_t target = static_cast<uint64_t>(capacity_ * low);
    std::vector<std::string> evicted;
    while (used_bytes_.load(std::memory_order_relaxed) > target) {
        Entry victim;
        std::string key;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = std::find_if(entries_.begin(), entries_.end(),
                                   [](const auto& item) {
                                       return item.second.local_owner;
                                   });
            if (it == entries_.end()) break;
            for (auto candidate = entries_.begin(); candidate != entries_.end();
                 ++candidate) {
                if (candidate->second.local_owner &&
                    candidate->second.sequence < it->second.sequence) {
                    it = candidate;
                }
            }
            key = it->first;
            victim = it->second;
        }

        if (handler) {
            auto result = handler({key});
            if (!result) return tl::make_unexpected(result.error());
        }
        auto free_result = Free(victim.allocation);
        if (!free_result) return tl::make_unexpected(free_result.error());

        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = entries_.find(key);
            if (it != entries_.end() && it->second.sequence == victim.sequence) {
                used_bytes_.fetch_sub(it->second.allocation.size,
                                      std::memory_order_relaxed);
                entries_.erase(it);
            }
        }
        evicted.push_back(key);
    }
    return evicted;
}

}  // namespace mooncake
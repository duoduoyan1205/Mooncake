#include "memory_pool_storage_backend.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "amdgpu_mpu_uapi.h"
#include "device/accelerator_registry.h"
#include "environ.h"

namespace mooncake {

namespace {
constexpr uint64_t kDefaultAlignment = 4096;
constexpr uint64_t kDefaultXferTimeoutNs = 30ULL * 1000 * 1000 * 1000;

uint64_t AlignUp(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

MemoryPoolStorageBackend::MemoryPoolStorageBackend(
    const FileStorageConfig& config)
    : StorageBackendInterface(config),
      device_path_(Environ::GetString("MOONCAKE_MEMORY_POOL_DEVICE",
                                     "/dev/amdgpu-mpu")),
      capacity_(static_cast<uint64_t>(std::max<int64_t>(
          0, config.total_size_limit))) {}

MemoryPoolStorageBackend::~MemoryPoolStorageBackend() {
    RemoveAll();
    CloseDevice();
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::OpenDevice() {
    if (fd_ >= 0) return {};

    fd_ = open(device_path_.c_str(), O_RDWR | O_CLOEXEC);
    if (fd_ < 0) {
        LOG(ERROR) << "Memory Pool: failed to open " << device_path_
                   << ": " << strerror(errno);
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }

    amdgpu_mpu::IoctlCaps caps{};
    if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_CAPS, &caps) == 0) {
        if (caps.mem_size != 0) {
            capacity_ = capacity_ == 0 ? caps.mem_size
                                       : std::min(capacity_, caps.mem_size);
        }
        LOG(INFO) << "Memory Pool: SUE v" << caps.sue_version
                  << ", queues=" << caps.num_queues
                  << ", mem_base=0x" << std::hex << caps.mem_base
                  << ", mem_size=" << std::dec << caps.mem_size;
    } else {
        // CAPS is a bring-up aid. Keep operating when an older driver does not
        // implement it; ALLOC/XFER are the required interfaces.
        VLOG(1) << "Memory Pool: CAPS ioctl unavailable: " << strerror(errno);
    }

    if (capacity_ == 0) {
        LOG(ERROR) << "Memory Pool: capacity is zero; set "
                      "MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES or expose "
                      "mem_size through the MPU driver";
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
    const uint64_t aligned_size = AlignUp(size, kDefaultAlignment);
    amdgpu_mpu::IoctlAlloc req{
        .size = aligned_size,
        .alignment = kDefaultAlignment,
        .handle = 0,
        .global_addr = 0,
    };
    if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_ALLOC, &req) < 0) {
        LOG(ERROR) << "Memory Pool: ALLOC failed, size=" << aligned_size
                   << ": " << strerror(errno);
        return tl::make_unexpected(ErrorCode::BUFFER_OVERFLOW);
    }
    return Allocation{req.handle, req.global_addr, aligned_size};
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::Free(
    const Allocation& allocation) {
    if (allocation.handle == 0) return {};
    amdgpu_mpu::IoctlFree req{.handle = allocation.handle};
    if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_FREE, &req) < 0) {
        LOG(ERROR) << "Memory Pool: FREE failed, handle=" << allocation.handle
                   << ": " << strerror(errno);
        return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    }
    return {};
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::TransferDeviceToPool(
    const Slice& source, const Allocation& destination, uint64_t offset) {
    amdgpu_mpu::IoctlXfer req{};
    req.src_addr = reinterpret_cast<uint64_t>(source.ptr);
    req.dst_addr = destination.global_addr + offset;
    req.length = source.size;
    req.path = amdgpu_mpu::kPathDToPoolWrite;
    req.flags = amdgpu_mpu::kXferSignal | amdgpu_mpu::kXferOrdered;
    req.fence_fd = -1;
    if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_XFER, &req) < 0) {
        return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    }

    if (req.cookie != 0) {
        amdgpu_mpu::IoctlWait wait{
            .cookie = req.cookie,
            .timeout_ns = kDefaultXferTimeoutNs,
            .status = 0,
            .reserved = 0,
        };
        if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_WAIT, &wait) < 0 ||
            wait.status != 0) {
            LOG(ERROR) << "Memory Pool: D->POOL transfer failed, cookie="
                       << req.cookie << ", status=" << wait.status;
            return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
        }
    }
    return {};
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::TransferPoolToDevice(
    const Allocation& source, const Slice& destination, uint64_t offset) {
    amdgpu_mpu::IoctlXfer req{};
    req.src_addr = source.global_addr + offset;
    req.dst_addr = reinterpret_cast<uint64_t>(destination.ptr);
    req.length = destination.size;
    req.path = amdgpu_mpu::kPathDToPoolRead;
    req.flags = amdgpu_mpu::kXferSignal | amdgpu_mpu::kXferOrdered;
    req.fence_fd = -1;
    if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_XFER, &req) < 0) {
        return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
    }

    if (req.cookie != 0) {
        amdgpu_mpu::IoctlWait wait{
            .cookie = req.cookie,
            .timeout_ns = kDefaultXferTimeoutNs,
            .status = 0,
            .reserved = 0,
        };
        if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_WAIT, &wait) < 0 ||
            wait.status != 0) {
            LOG(ERROR) << "Memory Pool: POOL->D transfer failed, cookie="
                       << req.cookie << ", status=" << wait.status;
            return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
        }
    }
    return {};
}

tl::expected<void*, ErrorCode> MemoryPoolStorageBackend::MapAllocation(
    const Allocation& allocation) {
    amdgpu_mpu::IoctlMmap req{
        .handle = allocation.handle,
        .offset = 0,
        .size = allocation.size,
        .flags = 0,
        .reserved = 0,
    };
    if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_MMAP, &req) < 0) {
        LOG(ERROR) << "Memory Pool: MMAP ioctl failed, handle="
                   << allocation.handle << ": " << strerror(errno);
        return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
    }

    // The kernel driver uses mmap's page-offset as {handle,page} where the
    // high 32 bits identify the allocation. Page zero maps the allocation.
    const off_t page_offset = static_cast<off_t>(allocation.handle) << 32;
    void* mapped = mmap(nullptr, allocation.size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd_, page_offset << 12);
    if (mapped == MAP_FAILED) {
        LOG(ERROR) << "Memory Pool: mmap failed, handle=" << allocation.handle
                   << ": " << strerror(errno);
        return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
    }
    return mapped;
}

void MemoryPoolStorageBackend::UnmapAllocation(void* address, uint64_t size) {
    if (address != nullptr && address != MAP_FAILED) {
        munmap(address, size);
    }
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::TransferWithCpuMapping(
    const Slice& source, const Allocation& allocation, bool to_pool,
    uint64_t offset) {
    auto map_res = MapAllocation(allocation);
    if (!map_res) return tl::make_unexpected(map_res.error());
    auto* base = static_cast<char*>(map_res.value()) + offset;
    if (to_pool) {
        std::memcpy(base, source.ptr, source.size);
    } else {
        std::memcpy(source.ptr, base, source.size);
    }
    UnmapAllocation(map_res.value(), allocation.size);
    return {};
}

bool MemoryPoolStorageBackend::LooksLikeDevicePointer(const void* ptr) const {
    if (ptr == nullptr) return false;
    auto& accelerators = device::GetAcceleratorRegistry().RuntimeAccelerators();
    device::PointerInfo info{};
    return accelerators.FindDeviceForPointer(const_cast<void*>(ptr), &info) !=
           nullptr;
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::EvictForSpace(
    uint64_t required_size, EvictionHandler eviction_handler) {
    if (required_size > capacity_) {
        return tl::make_unexpected(ErrorCode::KEYS_ULTRA_LIMIT);
    }

    while (used_bytes_.load(std::memory_order_relaxed) + required_size >
           capacity_) {
        std::string victim_key;
        Entry victim;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (entries_.empty()) {
                return tl::make_unexpected(ErrorCode::BUFFER_OVERFLOW);
            }
            auto victim_it = std::min_element(
                entries_.begin(), entries_.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.second.sequence < rhs.second.sequence;
                });
            victim_key = victim_it->first;
            victim = victim_it->second;
        }

        if (eviction_handler) {
            std::vector<std::string> keys{victim_key};
            auto notify = eviction_handler(keys);
            if (!notify) return tl::make_unexpected(notify.error());
        }

        auto free_res = Free(victim.allocation);
        if (!free_res) return free_res;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = entries_.find(victim_key);
            if (it != entries_.end() &&
                it->second.sequence == victim.sequence) {
                used_bytes_.fetch_sub(it->second.allocation.size,
                                      std::memory_order_relaxed);
                entries_.erase(it);
            }
        }
    }
    return {};
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::Init() {
    bool expected = false;
    if (!initialized_.compare_exchange_strong(expected, true,
                                               std::memory_order_acq_rel)) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    auto open_res = OpenDevice();
    if (!open_res) {
        initialized_.store(false, std::memory_order_release);
        return open_res;
    }
    LOG(INFO) << "Memory Pool backend initialized: device=" << device_path_
              << ", capacity=" << capacity_;
    return {};
}

tl::expected<int64_t, ErrorCode> MemoryPoolStorageBackend::BatchOffload(
    const std::unordered_map<std::string, std::vector<Slice>>& batch_object,
    std::function<ErrorCode(const std::vector<std::string>&,
                            std::vector<StorageObjectMetadata>&)> complete_handler,
    EvictionHandler eviction_handler) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    if (batch_object.empty()) {
        return tl::make_unexpected(ErrorCode::INVALID_KEY);
    }

    uint64_t batch_bytes = 0;
    for (const auto& [key, slices] : batch_object) {
        (void)key;
        for (const auto& slice : slices) batch_bytes += slice.size;
    }
    auto evict_res = EvictForSpace(batch_bytes, eviction_handler);
    if (!evict_res) return tl::make_unexpected(evict_res.error());

    std::vector<std::string> success_keys;
    std::vector<StorageObjectMetadata> metadatas;
    success_keys.reserve(batch_object.size());
    metadatas.reserve(batch_object.size());

    for (const auto& [key, slices] : batch_object) {
        uint64_t total = 0;
        for (const auto& slice : slices) total += slice.size;
        if (total == 0) continue;

        // Replacing an existing key first releases its old allocation. The
        // current master protocol treats the write as an overwrite of the
        // same LOCAL_DISK replica.
        Entry old_entry;
        bool had_old = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = entries_.find(key);
            if (it != entries_.end()) {
                old_entry = it->second;
                had_old = true;
            }
        }
        if (had_old) {
            auto free_res = Free(old_entry.allocation);
            if (!free_res) continue;
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = entries_.find(key);
            if (it != entries_.end() && it->second.sequence == old_entry.sequence) {
                used_bytes_.fetch_sub(it->second.allocation.size,
                                      std::memory_order_relaxed);
                entries_.erase(it);
            }
        }

        auto alloc_res = Allocate(total);
        if (!alloc_res) {
            continue;
        }
        Allocation allocation = alloc_res.value();
        uint64_t offset = 0;
        bool success = true;
        for (const auto& slice : slices) {
            if (LooksLikeDevicePointer(slice.ptr)) {
                auto xfer = TransferDeviceToPool(slice, allocation, offset);
                if (!xfer) {
                    success = false;
                    break;
                }
            } else {
                auto copy = TransferWithCpuMapping(slice, allocation, true, offset);
                if (!copy) {
                    success = false;
                    break;
                }
            }
            offset += slice.size;
        }
        if (!success) {
            (void)Free(allocation);
            continue;
        }

        const uint64_t seq = next_sequence_.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            entries_[key] = Entry{allocation, seq};
        }
        used_bytes_.fetch_add(allocation.size, std::memory_order_relaxed);
        success_keys.push_back(key);
        metadatas.push_back(StorageObjectMetadata{
            0, static_cast<int64_t>(allocation.global_addr), 0,
            static_cast<int64_t>(total), ""});
    }

    if (complete_handler && !success_keys.empty()) {
        auto ec = complete_handler(success_keys, metadatas);
        if (ec != ErrorCode::OK) return tl::make_unexpected(ec);
    }
    return static_cast<int64_t>(success_keys.size());
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::BatchLoad(
    std::unordered_map<std::string, Slice>& batched_slices) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }

    for (auto& [key, destination] : batched_slices) {
        Entry entry;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = entries_.find(key);
            if (it == entries_.end()) {
                return tl::make_unexpected(ErrorCode::OBJECT_NOT_FOUND);
            }
            entry = it->second;
        }
        if (destination.size > entry.allocation.size) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }

        if (LooksLikeDevicePointer(destination.ptr)) {
            auto xfer = TransferPoolToDevice(entry.allocation, destination, 0);
            if (!xfer) return xfer;
        } else {
            auto copy = TransferWithCpuMapping(destination, entry.allocation,
                                               false, 0);
            if (!copy) return copy;
        }
    }
    return {};
}

tl::expected<bool, ErrorCode> MemoryPoolStorageBackend::IsExist(
    const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.find(key) != entries_.end();
}

tl::expected<bool, ErrorCode> MemoryPoolStorageBackend::IsEnableOffloading() {
    if (!initialized_.load(std::memory_order_acquire)) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    return used_bytes_.load(std::memory_order_relaxed) < capacity_;
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::ScanMeta(
    const std::function<ErrorCode(const std::vector<std::string>&,
                                  std::vector<StorageObjectMetadata>&)>& handler) {
    std::vector<std::string> keys;
    std::vector<StorageObjectMetadata> metadatas;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        keys.reserve(entries_.size());
        metadatas.reserve(entries_.size());
        for (const auto& [key, entry] : entries_) {
            keys.push_back(key);
            metadatas.push_back(StorageObjectMetadata{
                0, static_cast<int64_t>(entry.allocation.global_addr), 0,
                static_cast<int64_t>(entry.allocation.size), ""});
        }
    }
    if (!keys.empty() && handler) {
        auto ec = handler(keys, metadatas);
        if (ec != ErrorCode::OK) return tl::make_unexpected(ec);
    }
    return {};
}

void MemoryPoolStorageBackend::RemoveAll() {
    if (fd_ < 0) return;
    std::vector<Allocation> allocations;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        allocations.reserve(entries_.size());
        for (const auto& [key, entry] : entries_) {
            (void)key;
            allocations.push_back(entry.allocation);
        }
        entries_.clear();
        used_bytes_.store(0, std::memory_order_relaxed);
    }
    for (const auto& allocation : allocations) {
        (void)Free(allocation);
    }
}

tl::expected<std::vector<std::string>, ErrorCode>
MemoryPoolStorageBackend::EvictAboveDiskWatermark(
    double high_watermark_ratio, double low_watermark_ratio,
    EvictionHandler eviction_handler) {
    if (capacity_ == 0 ||
        used_bytes_.load(std::memory_order_relaxed) <=
            static_cast<uint64_t>(capacity_ * high_watermark_ratio)) {
        return std::vector<std::string>{};
    }

    const uint64_t target =
        static_cast<uint64_t>(capacity_ * low_watermark_ratio);
    std::vector<std::string> evicted;
    while (used_bytes_.load(std::memory_order_relaxed) > target) {
        std::string victim_key;
        Entry victim;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (entries_.empty()) break;
            auto it = std::min_element(
                entries_.begin(), entries_.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.second.sequence < rhs.second.sequence;
                });
            victim_key = it->first;
            victim = it->second;
        }
        if (eviction_handler) {
            auto result = eviction_handler({victim_key});
            if (!result) return tl::make_unexpected(result.error());
        }
        auto free_res = Free(victim.allocation);
        if (!free_res) return tl::make_unexpected(free_res.error());
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(victim_key);
        if (it != entries_.end() && it->second.sequence == victim.sequence) {
            used_bytes_.fetch_sub(it->second.allocation.size,
                                  std::memory_order_relaxed);
            entries_.erase(it);
            evicted.push_back(victim_key);
        }
    }
    return evicted;
}

}  // namespace mooncake

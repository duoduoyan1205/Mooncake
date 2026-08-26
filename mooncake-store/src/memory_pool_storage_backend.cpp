#include "memory_pool_storage_backend.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "amdgpu_mpu_uapi.h"

namespace mooncake {

namespace {
constexpr uint64_t kDefaultAlignment = 4096;
constexpr int kFirstRenderNode = 128;
constexpr int kLastRenderNode = 191;

uint64_t AlignUp(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

std::string GetMemoryPoolDevicePath() {
    const char* value = std::getenv("MOONCAKE_MEMORY_POOL_DEVICE");
    return value != nullptr && value[0] != '\0' ? value : "";
}

}  // namespace

MemoryPoolStorageBackend::MemoryPoolStorageBackend(
    const FileStorageConfig& config)
    : StorageBackendInterface(config),
      device_path_(GetMemoryPoolDevicePath()),
      capacity_(static_cast<uint64_t>(std::max<int64_t>(
          0, config.total_size_limit))) {}

MemoryPoolStorageBackend::~MemoryPoolStorageBackend() {
    RemoveAll();
    CloseDevice();
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::OpenDevice() {
    if (fd_ >= 0) return {};

    std::vector<std::string> candidates;
    if (!device_path_.empty()) {
        candidates.push_back(device_path_);
    } else {
        for (int node = kFirstRenderNode; node <= kLastRenderNode; ++node) {
            candidates.push_back("/dev/dri/renderD" + std::to_string(node));
        }
        candidates.push_back("/dev/dri/card0");
        candidates.push_back("/dev/dri/card1");
    }

    for (const auto& path : candidates) {
        int candidate_fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
        if (candidate_fd < 0) continue;

        amdgpu_mpu::Caps caps{};
        if (ioctl(candidate_fd, MOONCAKE_AMDGPU_MPU_GET_CAPS, &caps) == 0 &&
            caps.version >= amdgpu_mpu::kUapiVersion &&
            (caps.flags & amdgpu_mpu::kCapP2p)) {
            fd_ = candidate_fd;
            device_path_ = path;
            if (caps.mem_size != 0) {
                capacity_ = capacity_ == 0 ? caps.mem_size
                                           : std::min(capacity_, caps.mem_size);
            }
            LOG(INFO) << "Memory Pool: opened MPU DRM node " << device_path_
                      << ", UAPI v" << caps.version << ", VMID=" << caps.vmid
                      << ", page_shift=" << caps.page_shift
                      << ", VA=[0x" << std::hex << caps.va_start << ", 0x"
                      << caps.va_end << "]"
                      << ", mem_base=0x" << caps.mem_base
                      << ", mem_size=" << std::dec << caps.mem_size;
            break;
        }
        close(candidate_fd);
    }

    if (fd_ < 0) {
        LOG(ERROR) << "Memory Pool: no compatible AMDGPU MPU DRM render node found";
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }

    if (capacity_ == 0) {
        LOG(ERROR) << "Memory Pool: capacity is zero; set "
                      "MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES or expose "
                      "mem_size through the MPU driver";
        CloseDevice();
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
    amdgpu_mpu::GemCreate req{};
    req.size = aligned_size;
    req.alignment = kDefaultAlignment;
    req.flags = amdgpu_mpu::kGemCreateNodeMemory;

    if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_GEM_CREATE, &req) < 0) {
        LOG(ERROR) << "Memory Pool: GEM_CREATE node-memory allocation failed, size="
                   << aligned_size << ": " << strerror(errno);
        return tl::make_unexpected(ErrorCode::BUFFER_OVERFLOW);
    }

    if (req.handle == 0 || req.target_addr == 0 || req.gpu_addr == 0) {
        LOG(ERROR) << "Memory Pool: GEM_CREATE returned incomplete allocation "
                   << "handle=" << req.handle << " gpu_addr=0x" << std::hex
                   << req.gpu_addr << " target_addr=0x" << req.target_addr;
        if (req.handle != 0) {
            amdgpu_mpu::GemClose close_req{.handle = req.handle, .pad = 0};
            (void)ioctl(fd_, MOONCAKE_AMDGPU_MPU_GEM_CLOSE, &close_req);
        }
        return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    }

    return Allocation{req.handle, req.target_addr, req.gpu_addr, aligned_size,
                      req.mmap_offset};
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::Free(
    const Allocation& allocation) {
    if (allocation.handle == 0) return {};
    amdgpu_mpu::GemClose req{.handle = allocation.handle, .pad = 0};
    if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_GEM_CLOSE, &req) < 0) {
        LOG(ERROR) << "Memory Pool: GEM_CLOSE failed, handle="
                   << allocation.handle << ": " << strerror(errno);
        return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    }
    return {};
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::TransferDeviceToPool(
    const Slice& source, const Allocation& destination, uint64_t offset) {
    (void)source;
    (void)destination;
    (void)offset;
    // The current MPU driver deliberately removed the old private XFER ioctl.
    // The next data-path patch must import/export the GEM as DMA-BUF and bind
    // it to the real GPU/NIC/SUE initiator path; do not silently fall back to
    // a CPU memcpy because node-memory BOs are not CPU-mappable.
    LOG(ERROR) << "Memory Pool: GPU->MPU transfer is not exposed by the current "
                  "DRM UAPI; DMA-BUF/SUE transfer integration is required";
    return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::TransferPoolToDevice(
    const Allocation& source, const Slice& destination, uint64_t offset) {
    (void)source;
    (void)destination;
    (void)offset;
    LOG(ERROR) << "Memory Pool: MPU->GPU transfer is not exposed by the current "
                  "DRM UAPI; DMA-BUF/SUE transfer integration is required";
    return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
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
            auto notify = eviction_handler({victim_key});
            if (!notify) return tl::make_unexpected(notify.error());
        }

        auto free_res = Free(victim.allocation);
        if (!free_res) return free_res;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(victim_key);
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
        if (!alloc_res) continue;
        Allocation allocation = alloc_res.value();

        uint64_t offset = 0;
        bool success = true;
        for (const auto& slice : slices) {
            auto xfer = TransferDeviceToPool(slice, allocation, offset);
            if (!xfer) {
                success = false;
                break;
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
            0, static_cast<int64_t>(allocation.target_addr), 0,
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
        auto xfer = TransferPoolToDevice(entry.allocation, destination, 0);
        if (!xfer) return xfer;
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
                0, static_cast<int64_t>(entry.allocation.target_addr), 0,
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

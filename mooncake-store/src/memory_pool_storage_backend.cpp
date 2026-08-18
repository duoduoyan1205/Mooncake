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
uint64_t AlignUp(uint64_t v) { return (v + kAlignment - 1) & ~(kAlignment - 1); }
std::string DevicePath() {
    const char* p = std::getenv("MOONCAKE_MEMORY_POOL_DEVICE");
    return p && *p ? p : "/dev/amdgpu-mpu";
}
}  // namespace

MemoryPoolStorageBackend::MemoryPoolStorageBackend(const FileStorageConfig& config)
    : StorageBackendInterface(config), device_path_(DevicePath()),
      capacity_(static_cast<uint64_t>(std::max<int64_t>(0, config.total_size_limit))) {}

MemoryPoolStorageBackend::~MemoryPoolStorageBackend() { RemoveAll(); CloseDevice(); }

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::OpenDevice() {
    if (fd_ >= 0) return {};
    fd_ = open(device_path_.c_str(), O_RDWR | O_CLOEXEC);
    if (fd_ < 0) return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    amdgpu_mpu::IoctlCaps caps{};
    if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_CAPS, &caps) == 0 && caps.mem_size)
        capacity_ = capacity_ ? std::min(capacity_, caps.mem_size) : caps.mem_size;
    if (!capacity_) {
        close(fd_); fd_ = -1;
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    return {};
}

void MemoryPoolStorageBackend::CloseDevice() { if (fd_ >= 0) { close(fd_); fd_ = -1; } }

tl::expected<MemoryPoolStorageBackend::Allocation, ErrorCode>
MemoryPoolStorageBackend::Allocate(uint64_t size) {
    amdgpu_mpu::IoctlAlloc req{AlignUp(size), kAlignment, 0, 0};
    if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_ALLOC, &req) < 0)
        return tl::make_unexpected(ErrorCode::BUFFER_OVERFLOW);
    return Allocation{req.handle, req.global_addr, req.size};
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::Free(const Allocation& a) {
    if (!a.handle) return {};
    amdgpu_mpu::IoctlFree req{a.handle};
    return ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_FREE, &req) < 0
               ? tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL) : tl::expected<void, ErrorCode>{};
}

bool MemoryPoolStorageBackend::LooksLikeDevicePointer(const void* ptr) const {
    if (!ptr) return false;
    auto& r = device::GetAcceleratorRegistry().RuntimeAccelerators();
    device::PointerInfo info{};
    return r.FindDeviceForPointer(const_cast<void*>(ptr), &info) != nullptr;
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::TransferGpuToGpu(
    const Slice& src, const Slice& dst, uint32_t path) {
    if (!LooksLikeDevicePointer(src.ptr) || !LooksLikeDevicePointer(dst.ptr))
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    if (!src.size || !dst.size || src.size != dst.size)
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    amdgpu_mpu::IoctlXfer req{};
    req.src_addr = reinterpret_cast<uint64_t>(src.ptr);
    req.dst_addr = reinterpret_cast<uint64_t>(dst.ptr);
    req.length = src.size;
    req.path = path;
    req.flags = amdgpu_mpu::kXferSignal | amdgpu_mpu::kXferOrdered;
    req.fence_fd = -1;
    if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_XFER, &req) < 0)
        return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    if (req.cookie) {
        amdgpu_mpu::IoctlWait wait{req.cookie, kTimeoutNs, 0, 0};
        if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_WAIT, &wait) < 0 || wait.status)
            return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
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
    const Slice& slice, const Allocation& a, uint64_t offset, bool to_pool) {
    if (!LooksLikeDevicePointer(slice.ptr) || offset + slice.size > a.size)
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    amdgpu_mpu::IoctlXfer req{};
    req.src_addr = to_pool ? reinterpret_cast<uint64_t>(slice.ptr) : a.global_addr + offset;
    req.dst_addr = to_pool ? a.global_addr + offset : reinterpret_cast<uint64_t>(slice.ptr);
    req.length = slice.size;
    req.path = to_pool ? amdgpu_mpu::kPathDToPoolWrite : amdgpu_mpu::kPathDToPoolRead;
    req.flags = amdgpu_mpu::kXferSignal | amdgpu_mpu::kXferOrdered;
    req.fence_fd = -1;
    if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_XFER, &req) < 0)
        return tl::make_unexpected(to_pool ? ErrorCode::FILE_WRITE_FAIL : ErrorCode::FILE_READ_FAIL);
    if (req.cookie) {
        amdgpu_mpu::IoctlWait wait{req.cookie, kTimeoutNs, 0, 0};
        if (ioctl(fd_, MOONCAKE_AMDGPU_MPU_IOCTL_WAIT, &wait) < 0 || wait.status)
            return tl::make_unexpected(to_pool ? ErrorCode::FILE_WRITE_FAIL : ErrorCode::FILE_READ_FAIL);
    }
    return {};
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::TransferDToPool(
    const Slice& src, uint64_t handle, uint64_t addr, uint64_t offset) {
    return Transfer(src, Allocation{handle, addr, offset + src.size}, offset, true);
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::TransferPoolToD(
    const Slice& dst, uint64_t handle, uint64_t addr, uint64_t offset) {
    return Transfer(dst, Allocation{handle, addr, offset + dst.size}, offset, false);
}

tl::expected<void*, ErrorCode> MemoryPoolStorageBackend::MapAllocation(const Allocation&) {
    return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
}
void MemoryPoolStorageBackend::UnmapAllocation(void*, uint64_t) {}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::EvictForSpace(
    uint64_t required, EvictionHandler handler) {
    while (used_bytes_.load() + required > capacity_) {
        Entry victim; std::string key;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (entries_.empty()) return tl::make_unexpected(ErrorCode::BUFFER_OVERFLOW);
            auto it = std::min_element(entries_.begin(), entries_.end(),
                [](const auto& a, const auto& b) { return a.second.sequence < b.second.sequence; });
            key = it->first; victim = it->second;
        }
        if (handler) { auto r = handler({key}); if (!r) return tl::make_unexpected(r.error()); }
        auto r = Free(victim.allocation); if (!r) return r;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(key);
        if (it != entries_.end() && it->second.sequence == victim.sequence) {
            used_bytes_ -= it->second.allocation.size; entries_.erase(it);
        }
    }
    return {};
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::Init() {
    bool expected = false;
    if (!initialized_.compare_exchange_strong(expected, true)) return {};
    auto r = OpenDevice();
    if (!r) initialized_.store(false);
    return r;
}

tl::expected<int64_t, ErrorCode> MemoryPoolStorageBackend::BatchOffload(
    const std::unordered_map<std::string, std::vector<Slice>>& objects,
    std::function<ErrorCode(const std::vector<std::string>&, std::vector<StorageObjectMetadata>&)> complete,
    EvictionHandler handler) {
    if (!initialized_) return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    uint64_t bytes = 0;
    for (const auto& [k, slices] : objects) for (const auto& s : slices) bytes += s.size;
    auto e = EvictForSpace(bytes, handler); if (!e) return tl::make_unexpected(e.error());
    std::vector<std::string> keys; std::vector<StorageObjectMetadata> meta;
    for (const auto& [key, slices] : objects) {
        uint64_t total = 0; for (const auto& s : slices) total += s.size;
        if (!total) continue;
        auto a = Allocate(total); if (!a) continue;
        uint64_t off = 0; bool ok = true;
        for (const auto& s : slices) { auto r = TransferDToPool(s, a.value().handle, a.value().global_addr, off); if (!r) { ok = false; break; } off += s.size; }
        if (!ok) { Free(a.value()); continue; }
        { std::lock_guard<std::mutex> lock(mutex_); entries_[key] = Entry{a.value(), next_sequence_++}; }
        used_bytes_ += a.value().size;
        keys.push_back(key);
        meta.push_back(StorageObjectMetadata{0, static_cast<int64_t>(a.value().global_addr), 0, static_cast<int64_t>(total), ""});
    }
    if (complete && !keys.empty()) { auto r = complete(keys, meta); if (r != ErrorCode::OK) return tl::make_unexpected(r); }
    return static_cast<int64_t>(keys.size());
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::BatchLoad(
    std::unordered_map<std::string, Slice>& objects) {
    if (!initialized_) return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    for (auto& [key, dst] : objects) {
        Entry e; { std::lock_guard<std::mutex> lock(mutex_); auto it = entries_.find(key); if (it == entries_.end()) return tl::make_unexpected(ErrorCode::OBJECT_NOT_FOUND); e = it->second; }
        if (dst.size > e.allocation.size) return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        auto r = TransferPoolToD(dst, e.allocation.handle, e.allocation.global_addr, 0); if (!r) return r;
    }
    return {};
}

tl::expected<bool, ErrorCode> MemoryPoolStorageBackend::IsExist(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_); return entries_.find(key) != entries_.end();
}
tl::expected<bool, ErrorCode> MemoryPoolStorageBackend::IsEnableOffloading() { return initialized_.load(); }

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::ScanMeta(
    const std::function<ErrorCode(const std::vector<std::string>&, std::vector<StorageObjectMetadata>&)>& handler) {
    std::vector<std::string> keys; std::vector<StorageObjectMetadata> meta;
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [k, e] : entries_) { keys.push_back(k); meta.push_back({0, static_cast<int64_t>(e.allocation.global_addr), 0, static_cast<int64_t>(e.allocation.size), ""}); }
    if (handler && !keys.empty()) { auto r = handler(keys, meta); if (r != ErrorCode::OK) return tl::make_unexpected(r); }
    return {};
}

void MemoryPoolStorageBackend::RemoveAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [k, e] : entries_) (void)Free(e.allocation);
    entries_.clear(); used_bytes_ = 0;
}

tl::expected<std::vector<std::string>, ErrorCode> MemoryPoolStorageBackend::EvictAboveDiskWatermark(
    double high, double low, EvictionHandler handler) {
    if (high <= 0 || low < 0 || low >= high) return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    uint64_t target = static_cast<uint64_t>(capacity_ * low);
    std::vector<std::string> evicted;
    while (used_bytes_.load() > target) {
        Entry victim; std::string key;
        { std::lock_guard<std::mutex> lock(mutex_); if (entries_.empty()) break; auto it = std::min_element(entries_.begin(), entries_.end(), [](const auto& a, const auto& b){ return a.second.sequence < b.second.sequence; }); key = it->first; victim = it->second; }
        if (handler) { auto r = handler({key}); if (!r) return tl::make_unexpected(r.error()); }
        auto r = Free(victim.allocation); if (!r) return tl::make_unexpected(r.error());
        { std::lock_guard<std::mutex> lock(mutex_); auto it = entries_.find(key); if (it != entries_.end()) { used_bytes_ -= it->second.allocation.size; entries_.erase(it); } }
        evicted.push_back(key);
    }
    return evicted;
}

}  // namespace mooncake

#include "memory_pool_storage_backend.h"

#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <unordered_map>

#include "device/accelerator_registry.h"

namespace mooncake {
namespace {
constexpr uint64_t kAlignment = 4096;
constexpr uint64_t kTimeoutNs = 30ULL * 1000 * 1000 * 1000;

// ABI-compatible subset of the userspace MPU/SUE verbs library.  The
// Memory Pool backend deliberately loads this C ABI at runtime so Mooncake
// does not need to link to a vendor-specific library at build time.
struct amdgpu_mpu_ctx;
struct amdgpu_mpu_buf {
    uint64_t handle;
    uint64_t global_addr;
    uint64_t size;
    void* cpu_addr;
    size_t mapped_len;
};
struct amdgpu_mpu_caps {
    uint32_t sue_version;
    uint32_t sue_caps;
    uint32_t num_queues;
    uint32_t flags;
    uint64_t mem_base;
    uint64_t mem_size;
};

enum amdgpu_mpu_path {
    kPToD = 0,
    kDToP = 1,
    kDToPoolWrite = 2,
    kDToPoolRead = 3,
};

using FnOpen = int (*)(const char*, amdgpu_mpu_ctx**);
using FnClose = void (*)(amdgpu_mpu_ctx*);
using FnGetCaps = int (*)(amdgpu_mpu_ctx*, amdgpu_mpu_caps*);
using FnAlloc = int (*)(amdgpu_mpu_ctx*, size_t, size_t, amdgpu_mpu_buf*);
using FnFree = int (*)(amdgpu_mpu_ctx*, amdgpu_mpu_buf*);
using FnSubmitAndWait = int (*)(amdgpu_mpu_ctx*, amdgpu_mpu_path, uint64_t,
                                uint64_t, size_t, uint32_t, uint64_t, int*);

struct SueVerbsApi {
    FnOpen open = nullptr;
    FnClose close = nullptr;
    FnGetCaps get_caps = nullptr;
    FnAlloc alloc = nullptr;
    FnFree free = nullptr;
    FnSubmitAndWait submit_and_wait = nullptr;
};

std::mutex& ApiMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<void*, SueVerbsApi>& ApiMap() {
    static std::unordered_map<void*, SueVerbsApi> map;
    return map;
}

bool GetApi(void* ctx, SueVerbsApi* api) {
    std::lock_guard<std::mutex> lock(ApiMutex());
    auto it = ApiMap().find(ctx);
    if (it == ApiMap().end()) return false;
    *api = it->second;
    return true;
}

uint64_t AlignUp(uint64_t v) {
    return (v + kAlignment - 1) & ~(kAlignment - 1);
}

std::string DevicePath() {
    const char* p = std::getenv("MOONCAKE_MEMORY_POOL_DEVICE");
    return p && *p ? p : "/dev/amdgpu-mpu";
}

std::string SueVerbsLibrary() {
    const char* p = std::getenv("MOONCAKE_SUEVERBS_LIBRARY");
    return p && *p ? p : "libsueverbs.so";
}

std::string EnvString(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

template <typename T>
bool LoadSymbol(void* handle, const char* name, T* out) {
    dlerror();
    void* symbol = dlsym(handle, name);
    const char* error = dlerror();
    if (error || !symbol) return false;
    *out = reinterpret_cast<T>(symbol);
    return true;
}
}  // namespace

MemoryPoolStorageBackend::MemoryPoolStorageBackend(
    const FileStorageConfig& config)
    : StorageBackendInterface(config),
      sueverbs_library_(SueVerbsLibrary()),
      device_path_(DevicePath()),
      capacity_(static_cast<uint64_t>(
          std::max<int64_t>(0, config.total_size_limit))) {}

MemoryPoolStorageBackend::~MemoryPoolStorageBackend() {
    RemoveAll();
    CloseDevice();
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::OpenDevice() {
    if (sueverbs_ctx_) return {};

    sueverbs_handle_ = dlopen(sueverbs_library_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!sueverbs_handle_) {
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }

    SueVerbsApi api;
    const bool loaded =
        LoadSymbol(sueverbs_handle_, "amdgpu_mpu_open", &api.open) &&
        LoadSymbol(sueverbs_handle_, "amdgpu_mpu_close", &api.close) &&
        LoadSymbol(sueverbs_handle_, "amdgpu_mpu_get_caps", &api.get_caps) &&
        LoadSymbol(sueverbs_handle_, "amdgpu_mpu_alloc", &api.alloc) &&
        LoadSymbol(sueverbs_handle_, "amdgpu_mpu_free", &api.free) &&
        LoadSymbol(sueverbs_handle_, "amdgpu_mpu_submit_and_wait",
                   &api.submit_and_wait);
    if (!loaded) {
        dlclose(sueverbs_handle_);
        sueverbs_handle_ = nullptr;
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }

    amdgpu_mpu_ctx* ctx = nullptr;
    if (api.open(device_path_.c_str(), &ctx) != 0 || !ctx) {
        dlclose(sueverbs_handle_);
        sueverbs_handle_ = nullptr;
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }

    amdgpu_mpu_caps caps{};
    if (api.get_caps(ctx, &caps) != 0 || !caps.mem_size) {
        api.close(ctx);
        dlclose(sueverbs_handle_);
        sueverbs_handle_ = nullptr;
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    capacity_ = capacity_ ? std::min(capacity_, caps.mem_size) : caps.mem_size;
    if (!capacity_) {
        api.close(ctx);
        dlclose(sueverbs_handle_);
        sueverbs_handle_ = nullptr;
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    {
        std::lock_guard<std::mutex> lock(ApiMutex());
        ApiMap()[ctx] = api;
    }
    sueverbs_ctx_ = ctx;
    return {};
}

void MemoryPoolStorageBackend::CloseDevice() {
    if (!sueverbs_handle_) return;

    SueVerbsApi api;
    if (sueverbs_ctx_ && GetApi(sueverbs_ctx_, &api)) {
        api.close(static_cast<amdgpu_mpu_ctx*>(sueverbs_ctx_));
        std::lock_guard<std::mutex> lock(ApiMutex());
        ApiMap().erase(sueverbs_ctx_);
    }
    sueverbs_ctx_ = nullptr;
    dlclose(sueverbs_handle_);
    sueverbs_handle_ = nullptr;
}

tl::expected<MemoryPoolStorageBackend::Allocation, ErrorCode>
MemoryPoolStorageBackend::Allocate(uint64_t size) {
    if (!sueverbs_ctx_) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    SueVerbsApi api;
    if (!GetApi(sueverbs_ctx_, &api)) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }

    amdgpu_mpu_buf buf{};
    if (api.alloc(static_cast<amdgpu_mpu_ctx*>(sueverbs_ctx_), AlignUp(size),
                  kAlignment, &buf) != 0) {
        return tl::make_unexpected(ErrorCode::BUFFER_OVERFLOW);
    }
    return Allocation{buf.handle, buf.global_addr, buf.size};
}

tl::expected<void, ErrorCode> MemoryPoolStorageBackend::Free(
    const Allocation& allocation) {
    if (!allocation.handle || !sueverbs_ctx_) return {};
    SueVerbsApi api;
    if (!GetApi(sueverbs_ctx_, &api)) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }

    amdgpu_mpu_buf buf{};
    buf.handle = allocation.handle;
    buf.global_addr = allocation.global_addr;
    buf.size = allocation.size;
    if (api.free(static_cast<amdgpu_mpu_ctx*>(sueverbs_ctx_), &buf) != 0) {
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
    if (!sueverbs_ctx_ || !LooksLikeDevicePointer(src.ptr) ||
        !LooksLikeDevicePointer(dst.ptr) || !src.size || !dst.size ||
        src.size != dst.size) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    SueVerbsApi api;
    if (!GetApi(sueverbs_ctx_, &api)) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }

    int status = 0;
    const uint32_t flags = amdgpu_mpu::kXferSignal | amdgpu_mpu::kXferOrdered;
    const int rc = api.submit_and_wait(
        static_cast<amdgpu_mpu_ctx*>(sueverbs_ctx_),
        static_cast<amdgpu_mpu_path>(path), reinterpret_cast<uint64_t>(src.ptr),
        reinterpret_cast<uint64_t>(dst.ptr), src.size, flags, kTimeoutNs,
        &status);
    if (rc != 0 || status != 0) {
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
    const Slice& slice, const Allocation& allocation, uint64_t offset,
    bool to_pool) {
    if (!sueverbs_ctx_ || !LooksLikeDevicePointer(slice.ptr) || !slice.size ||
        offset > allocation.size || slice.size > allocation.size - offset) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    SueVerbsApi api;
    if (!GetApi(sueverbs_ctx_, &api)) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }

    const uint64_t pool_addr = allocation.global_addr + offset;
    const uint32_t flags = amdgpu_mpu::kXferSignal | amdgpu_mpu::kXferOrdered;
    const uint64_t src_addr = to_pool ? reinterpret_cast<uint64_t>(slice.ptr)
                                      : pool_addr;
    const uint64_t dst_addr = to_pool ? pool_addr
                                      : reinterpret_cast<uint64_t>(slice.ptr);
    const uint32_t path = to_pool ? kDToPoolWrite : kDToPoolRead;

    int status = 0;
    const int rc = api.submit_and_wait(
        static_cast<amdgpu_mpu_ctx*>(sueverbs_ctx_),
        static_cast<amdgpu_mpu_path>(path), src_addr, dst_addr, slice.size,
        flags, kTimeoutNs, &status);
    if (rc != 0 || status != 0) {
        return tl::make_unexpected(to_pool ? ErrorCode::FILE_WRITE_FAIL
                                           : ErrorCode::FILE_READ_FAIL);
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
    if (!eviction_result) return tl::make_unexpected(eviction_result.error());

    std::vector<std::string> keys;
    std::vector<StorageObjectMetadata> metadata;
    const std::string endpoint =
        EnvString("MOONCAKE_MEMORY_POOL_TRANSPORT_ENDPOINT");

    for (const auto& [key, slices] : objects) {
        uint64_t total = 0;
        for (const auto& slice : slices) total += slice.size;
        if (!total) continue;

        auto allocation = Allocate(total);
        if (!allocation) continue;

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
        metadata.push_back(StorageObjectMetadata{
            static_cast<int64_t>(allocation.value().handle),
            static_cast<int64_t>(allocation.value().global_addr), 0,
            static_cast<int64_t>(total), endpoint});
    }

    if (complete && !keys.empty()) {
        auto result = complete(keys, metadata);
        if (result != ErrorCode::OK) return tl::make_unexpected(result);
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
    const std::string endpoint =
        EnvString("MOONCAKE_MEMORY_POOL_TRANSPORT_ENDPOINT");
    {
        std::lock_guard<std::mutex> lock(mutex_);
        keys.reserve(entries_.size());
        metadata.reserve(entries_.size());
        for (const auto& [key, entry] : entries_) {
            keys.push_back(key);
            metadata.push_back(StorageObjectMetadata{
                static_cast<int64_t>(entry.allocation.handle),
                static_cast<int64_t>(entry.allocation.global_addr), 0,
                static_cast<int64_t>(entry.allocation.size), endpoint});
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
#include "memory_pool_transfer_engine.h"

#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <cerrno>
#include <string>

#include "device/accelerator_registry.h"

namespace mooncake {
namespace {
constexpr uint64_t kAlignment = 4096;
constexpr uint64_t kTimeoutNs = 30ULL * 1000 * 1000 * 1000;
constexpr uint32_t kXferSignal = 1u << 0;
constexpr uint32_t kXferOrdered = 1u << 3;

enum amdgpu_mpu_path {
    AMDGPU_MPU_P_TO_D = 0,
    AMDGPU_MPU_D_TO_P = 1,
    AMDGPU_MPU_D_TO_POOL_WRITE = 2,
    AMDGPU_MPU_D_TO_POOL_READ = 3,
};

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

using FnOpen = int (*)(const char*, amdgpu_mpu_ctx**);
using FnClose = void (*)(amdgpu_mpu_ctx*);
using FnGetCaps = int (*)(amdgpu_mpu_ctx*, amdgpu_mpu_caps*);
using FnAlloc = int (*)(amdgpu_mpu_ctx*, size_t, size_t, amdgpu_mpu_buf*);
using FnFree = int (*)(amdgpu_mpu_ctx*, amdgpu_mpu_buf*);
using FnSubmitAndWait = int (*)(amdgpu_mpu_ctx*, amdgpu_mpu_path, uint64_t,
                                uint64_t, size_t, uint32_t, uint64_t, int*);

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

struct MemoryPoolTransferEngine::Context {};

struct MemoryPoolTransferEngine::Api {
    FnOpen open = nullptr;
    FnClose close = nullptr;
    FnGetCaps get_caps = nullptr;
    FnAlloc alloc = nullptr;
    FnFree free = nullptr;
    FnSubmitAndWait submit_and_wait = nullptr;
};

MemoryPoolTransferEngine::MemoryPoolTransferEngine(
    std::string sueverbs_library, std::string device_path)
    : sueverbs_library_(std::move(sueverbs_library)),
      device_path_(std::move(device_path)),
      api_(std::make_unique<Api>()) {}

MemoryPoolTransferEngine::~MemoryPoolTransferEngine() { Close(); }

int MemoryPoolTransferEngine::LoadApi() {
    if (!library_handle_) return -EINVAL;
    return LoadSymbol(library_handle_, "amdgpu_mpu_open", &api_->open) &&
                   LoadSymbol(library_handle_, "amdgpu_mpu_close", &api_->close) &&
                   LoadSymbol(library_handle_, "amdgpu_mpu_get_caps", &api_->get_caps) &&
                   LoadSymbol(library_handle_, "amdgpu_mpu_alloc", &api_->alloc) &&
                   LoadSymbol(library_handle_, "amdgpu_mpu_free", &api_->free) &&
                   LoadSymbol(library_handle_, "amdgpu_mpu_submit_and_wait",
                              &api_->submit_and_wait)
               ? 0
               : -ENOSYS;
}

void MemoryPoolTransferEngine::UnloadApi() {
    if (library_handle_) {
        dlclose(library_handle_);
        library_handle_ = nullptr;
    }
    api_ = std::make_unique<Api>();
}

int MemoryPoolTransferEngine::Open() {
    if (ctx_) return 0;

    library_handle_ = dlopen(sueverbs_library_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_handle_) return -ENOENT;

    int rc = LoadApi();
    if (rc) {
        UnloadApi();
        return rc;
    }

    amdgpu_mpu_ctx* raw_ctx = nullptr;
    rc = api_->open(device_path_.c_str(), &raw_ctx);
    if (rc || !raw_ctx) {
        UnloadApi();
        return rc ? rc : -ENODEV;
    }

    amdgpu_mpu_caps caps{};
    rc = api_->get_caps(raw_ctx, &caps);
    if (rc || !caps.mem_size) {
        api_->close(raw_ctx);
        UnloadApi();
        return rc ? rc : -ENODEV;
    }

    capacity_ = caps.mem_size;
    ctx_ = reinterpret_cast<Context*>(raw_ctx);
    return 0;
}

void MemoryPoolTransferEngine::Close() {
    if (ctx_ && api_ && api_->close) {
        api_->close(reinterpret_cast<amdgpu_mpu_ctx*>(ctx_));
    }
    ctx_ = nullptr;
    capacity_ = 0;
    UnloadApi();
}

int MemoryPoolTransferEngine::Allocate(uint64_t size, Allocation* allocation) {
    if (!ctx_ || !allocation || !size) return -EINVAL;
    amdgpu_mpu_buf buf{};
    const uint64_t aligned = (size + kAlignment - 1) & ~(kAlignment - 1);
    int rc = api_->alloc(reinterpret_cast<amdgpu_mpu_ctx*>(ctx_), aligned,
                         kAlignment, &buf);
    if (rc) return rc;
    allocation->handle = buf.handle;
    allocation->global_addr = buf.global_addr;
    allocation->size = buf.size;
    return 0;
}

int MemoryPoolTransferEngine::Free(const Allocation& allocation) {
    if (!allocation.handle || !ctx_) return 0;
    amdgpu_mpu_buf buf{};
    buf.handle = allocation.handle;
    buf.global_addr = allocation.global_addr;
    buf.size = allocation.size;
    return api_->free(reinterpret_cast<amdgpu_mpu_ctx*>(ctx_), &buf);
}

int MemoryPoolTransferEngine::Transfer(AccessPath path, uint64_t source_addr,
                                       uint64_t target_addr, size_t length) {
    if (!ctx_ || !length) return -EINVAL;
    int status = 0;
    return api_->submit_and_wait(
        reinterpret_cast<amdgpu_mpu_ctx*>(ctx_),
        static_cast<amdgpu_mpu_path>(static_cast<uint32_t>(path)), source_addr,
        target_addr, length, kXferSignal | kXferOrdered, kTimeoutNs, &status) == 0
               ? status
               : -EIO;
}

int MemoryPoolTransferEngine::TransferPToD(uint64_t source_addr,
                                           uint64_t target_addr, size_t length) {
    return Transfer(AccessPath::kPToD, source_addr, target_addr, length);
}

int MemoryPoolTransferEngine::TransferDToP(uint64_t source_addr,
                                           uint64_t target_addr, size_t length) {
    return Transfer(AccessPath::kDToP, source_addr, target_addr, length);
}

int MemoryPoolTransferEngine::TransferDToPool(uint64_t source_addr,
                                              uint64_t pool_addr, size_t length) {
    return Transfer(AccessPath::kDToPool, source_addr, pool_addr, length);
}

int MemoryPoolTransferEngine::TransferPoolToD(uint64_t pool_addr,
                                              uint64_t target_addr, size_t length) {
    return Transfer(AccessPath::kPoolToD, pool_addr, target_addr, length);
}

bool MemoryPoolTransferEngine::LooksLikeDevicePointer(const void* ptr) const {
    if (!ptr) return false;
    auto& registry = device::GetAcceleratorRegistry().RuntimeAccelerators();
    device::PointerInfo info{};
    return registry.FindDeviceForPointer(const_cast<void*>(ptr), &info) != nullptr;
}

}  // namespace mooncake

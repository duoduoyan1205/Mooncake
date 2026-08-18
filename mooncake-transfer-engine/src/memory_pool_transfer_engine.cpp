#include "memory_pool_transfer_engine.h"

#include <cerrno>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <libamdgpu_mpu.h>

namespace mooncake {
namespace {
constexpr uint64_t kAlignment = 4096;
constexpr uint64_t kTimeoutNs = 30ULL * 1000 * 1000 * 1000;

using FnOpen = int (*)(const char*, amdgpu_mpu_ctx_t**);
using FnClose = void (*)(amdgpu_mpu_ctx_t*);
using FnGetCaps = int (*)(amdgpu_mpu_ctx_t*, amdgpu_mpu_caps_t*);
using FnAlloc = int (*)(amdgpu_mpu_ctx_t*, size_t, size_t, amdgpu_mpu_buf_t*);
using FnFree = int (*)(amdgpu_mpu_ctx_t*, amdgpu_mpu_buf_t*);
using FnSubmitAndWait = int (*)(amdgpu_mpu_ctx_t*, amdgpu_mpu_path_t, uint64_t,
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

std::vector<std::string> SplitDevices(const std::string& devices) {
    std::vector<std::string> result;
    std::stringstream stream(devices);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const auto first = item.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        const auto last = item.find_last_not_of(" \t\r\n");
        result.push_back(item.substr(first, last - first + 1));
    }
    return result;
}
}  // namespace

struct MemoryPoolTransferEngine::Api {
    FnOpen open = nullptr;
    FnClose close = nullptr;
    FnGetCaps get_caps = nullptr;
    FnAlloc alloc = nullptr;
    FnFree free = nullptr;
    FnSubmitAndWait submit_and_wait = nullptr;
};

struct MemoryPoolTransferEngine::Node {
    std::string device_path;
    amdgpu_mpu_ctx_t* ctx = nullptr;
    uint64_t capacity = 0;
};

struct MemoryPoolTransferEngine::Context {
    explicit Context(std::string library) : library_path(std::move(library)) {}

    std::string library_path;
    void* library_handle = nullptr;
    std::unique_ptr<Api> api = std::make_unique<Api>();
    std::vector<Node> nodes;
    uint64_t next_node = 0;
    mutable std::mutex mutex;
};

MemoryPoolTransferEngine::MemoryPoolTransferEngine(
    std::string sueverbs_library, std::string device_paths)
    : context_(std::make_unique<Context>(std::move(sueverbs_library))) {
    for (const auto& device : SplitDevices(device_paths)) {
        Node node;
        node.device_path = device;
        context_->nodes.push_back(std::move(node));
    }
}

MemoryPoolTransferEngine::~MemoryPoolTransferEngine() { Close(); }

int MemoryPoolTransferEngine::Open() {
    if (!context_ || context_->nodes.empty()) return -EINVAL;
    if (IsOpen()) return 0;
    if (context_->library_path.empty()) return -EINVAL;

    context_->library_handle =
        dlopen(context_->library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!context_->library_handle) return -ENOENT;

    auto& api = *context_->api;
    const bool loaded =
        LoadSymbol(context_->library_handle, "amdgpu_mpu_open", &api.open) &&
        LoadSymbol(context_->library_handle, "amdgpu_mpu_close", &api.close) &&
        LoadSymbol(context_->library_handle, "amdgpu_mpu_get_caps", &api.get_caps) &&
        LoadSymbol(context_->library_handle, "amdgpu_mpu_alloc", &api.alloc) &&
        LoadSymbol(context_->library_handle, "amdgpu_mpu_free", &api.free) &&
        LoadSymbol(context_->library_handle, "amdgpu_mpu_submit_and_wait",
                   &api.submit_and_wait);
    if (!loaded) {
        Close();
        return -ENOSYS;
    }

    for (auto& node : context_->nodes) {
        amdgpu_mpu_ctx_t* raw_ctx = nullptr;
        int rc = api.open(node.device_path.c_str(), &raw_ctx);
        if (rc || !raw_ctx) {
            Close();
            return rc ? rc : -ENODEV;
        }

        amdgpu_mpu_caps_t caps{};
        rc = api.get_caps(raw_ctx, &caps);
        if (rc || !caps.mem_size) {
            api.close(raw_ctx);
            Close();
            return rc ? rc : -ENODEV;
        }
        node.ctx = raw_ctx;
        node.capacity = caps.mem_size;
    }
    return 0;
}

void MemoryPoolTransferEngine::Close() {
    if (!context_) return;
    if (context_->api && context_->api->close) {
        for (auto& node : context_->nodes) {
            if (node.ctx) context_->api->close(node.ctx);
            node.ctx = nullptr;
            node.capacity = 0;
        }
    }
    if (context_->library_handle) dlclose(context_->library_handle);
    context_->library_handle = nullptr;
    context_->next_node = 0;
    context_->api = std::make_unique<Api>();
}

bool MemoryPoolTransferEngine::IsOpen() const {
    if (!context_ || !context_->library_handle || context_->nodes.empty())
        return false;
    for (const auto& node : context_->nodes) {
        if (!node.ctx) return false;
    }
    return true;
}

size_t MemoryPoolTransferEngine::NodeCount() const {
    return context_ ? context_->nodes.size() : 0;
}

uint64_t MemoryPoolTransferEngine::NodeCapacity(uint32_t node_id) const {
    if (!context_ || node_id >= context_->nodes.size()) return 0;
    return context_->nodes[node_id].capacity;
}

uint64_t MemoryPoolTransferEngine::Capacity() const {
    if (!context_) return 0;
    uint64_t total = 0;
    for (const auto& node : context_->nodes) {
        if (std::numeric_limits<uint64_t>::max() - total < node.capacity)
            return std::numeric_limits<uint64_t>::max();
        total += node.capacity;
    }
    return total;
}

int MemoryPoolTransferEngine::Allocate(uint64_t size, Allocation* allocation) {
    if (!IsOpen() || !allocation || !size) return -EINVAL;
    const uint64_t aligned = (size + kAlignment - 1) & ~(kAlignment - 1);

    std::lock_guard<std::mutex> lock(context_->mutex);
    const uint32_t node_id = static_cast<uint32_t>(
        context_->next_node++ % context_->nodes.size());
    Node& node = context_->nodes[node_id];

    amdgpu_mpu_buf_t buf{};
    const int rc = context_->api->alloc(node.ctx, aligned, kAlignment, &buf);
    if (rc) return rc;

    allocation->node_id = node_id;
    allocation->handle = buf.handle;
    allocation->global_addr = buf.global_addr;
    allocation->size = buf.size;
    return 0;
}

int MemoryPoolTransferEngine::Free(const Allocation& allocation) {
    if (!IsOpen() || !allocation.handle) return 0;
    if (allocation.node_id >= context_->nodes.size()) return -EINVAL;
    Node& node = context_->nodes[allocation.node_id];
    amdgpu_mpu_buf_t buf{};
    buf.handle = allocation.handle;
    buf.global_addr = allocation.global_addr;
    buf.size = allocation.size;
    return context_->api->free(node.ctx, &buf);
}

int MemoryPoolTransferEngine::Transfer(AccessPath path, uint32_t node_id,
                                       uint64_t source_addr, uint64_t target_addr,
                                       size_t length) {
    if (!IsOpen() || !length || node_id >= context_->nodes.size()) return -EINVAL;
    Node& node = context_->nodes[node_id];
    int status = 0;
    const int rc = context_->api->submit_and_wait(
        node.ctx, static_cast<amdgpu_mpu_path_t>(static_cast<uint32_t>(path)),
        source_addr, target_addr, length,
        AMDGPU_MPU_XFER_F_SIGNAL | AMDGPU_MPU_XFER_F_ORDERED, kTimeoutNs,
        &status);
    return rc == 0 ? status : rc;
}

int MemoryPoolTransferEngine::TransferPToD(uint32_t node_id,
                                           uint64_t source_addr,
                                           uint64_t target_addr, size_t length) {
    return Transfer(AccessPath::kPToD, node_id, source_addr, target_addr, length);
}

int MemoryPoolTransferEngine::TransferDToP(uint32_t node_id,
                                           uint64_t source_addr,
                                           uint64_t target_addr, size_t length) {
    return Transfer(AccessPath::kDToP, node_id, source_addr, target_addr, length);
}

int MemoryPoolTransferEngine::TransferDToPool(
    const Allocation& allocation, uint64_t source_addr, size_t length,
    uint64_t offset) {
    if (offset > allocation.size || length > allocation.size - offset)
        return -EINVAL;
    return Transfer(AccessPath::kDToPool, allocation.node_id, source_addr,
                    allocation.global_addr + offset, length);
}

int MemoryPoolTransferEngine::TransferPoolToD(
    const Allocation& allocation, uint64_t target_addr, size_t length,
    uint64_t offset) {
    if (offset > allocation.size || length > allocation.size - offset)
        return -EINVAL;
    return Transfer(AccessPath::kPoolToD, allocation.node_id,
                    allocation.global_addr + offset, target_addr, length);
}

}  // namespace mooncake

#include "memory_pool_transfer_engine.h"

#include <cerrno>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace mooncake {
namespace {
constexpr uint64_t kAlignment = 4096;

using FnOpen = int (*)(const char*, amdgpu_mpu_ctx_t**);
using FnClose = void (*)(amdgpu_mpu_ctx_t*);
using FnGetCaps = int (*)(amdgpu_mpu_ctx_t*, amdgpu_mpu_caps_t*);
using FnAllocNode = int (*)(amdgpu_mpu_ctx_t*, size_t, size_t, amdgpu_mpu_buf_t*);
using FnFree = int (*)(amdgpu_mpu_ctx_t*, amdgpu_mpu_buf_t*);
using FnExportDmaBuf = int (*)(amdgpu_mpu_ctx_t*, const amdgpu_mpu_buf_t*, int, int*);
using FnTargetRange = int (*)(const amdgpu_mpu_buf_t*, uint64_t, size_t, uint64_t*);
using FnBoMap = int (*)(amdgpu_mpu_ctx_t*, amdgpu_mpu_buf_t*, size_t, size_t);
using FnBoUnmap = int (*)(amdgpu_mpu_buf_t*, size_t);

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

MemoryPoolTransferEngine::Allocation::~Allocation() {
    if (dmabuf_fd >= 0) {
        close(dmabuf_fd);
        dmabuf_fd = -1;
    }
}

MemoryPoolTransferEngine::Allocation::Allocation(Allocation&& other) noexcept
    : node_id(other.node_id), buf(other.buf), dmabuf_fd(other.dmabuf_fd) {
    other.node_id = 0;
    other.buf = {};
    other.dmabuf_fd = -1;
}

MemoryPoolTransferEngine::Allocation&
MemoryPoolTransferEngine::Allocation::operator=(Allocation&& other) noexcept {
    if (this == &other) return *this;
    if (dmabuf_fd >= 0) close(dmabuf_fd);
    node_id = other.node_id;
    buf = other.buf;
    dmabuf_fd = other.dmabuf_fd;
    other.node_id = 0;
    other.buf = {};
    other.dmabuf_fd = -1;
    return *this;
}

struct MemoryPoolTransferEngine::Api {
    FnOpen open = nullptr;
    FnClose close = nullptr;
    FnGetCaps get_caps = nullptr;
    FnAllocNode alloc_node = nullptr;
    FnFree free = nullptr;
    FnExportDmaBuf export_dmabuf = nullptr;
    FnTargetRange target_range = nullptr;
    FnBoMap bo_map = nullptr;
    FnBoUnmap bo_unmap = nullptr;
};

struct MemoryPoolTransferEngine::Node {
    std::string device_path;
    amdgpu_mpu_ctx_t* ctx = nullptr;
    uint64_t capacity = 0;
    amdgpu_mpu_caps_t caps{};
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
        LoadSymbol(context_->library_handle, "amdgpu_mpu_alloc_node", &api.alloc_node) &&
        LoadSymbol(context_->library_handle, "amdgpu_mpu_free", &api.free) &&
        LoadSymbol(context_->library_handle, "amdgpu_mpu_bo_export_dmabuf", &api.export_dmabuf) &&
        LoadSymbol(context_->library_handle, "amdgpu_mpu_target_range", &api.target_range) &&
        LoadSymbol(context_->library_handle, "amdgpu_mpu_bo_map", &api.bo_map) &&
        LoadSymbol(context_->library_handle, "amdgpu_mpu_bo_unmap", &api.bo_unmap);
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
        node.caps = caps;
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
            node.caps = {};
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
    if (allocation->valid()) return -EBUSY;
    if (size > std::numeric_limits<uint64_t>::max() - (kAlignment - 1))
        return -EINVAL;
    const uint64_t aligned = (size + kAlignment - 1) & ~(kAlignment - 1);

    std::lock_guard<std::mutex> lock(context_->mutex);
    const uint32_t node_id = static_cast<uint32_t>(
        context_->next_node++ % context_->nodes.size());
    Node& node = context_->nodes[node_id];

    allocation->buf = {};
    const int rc = context_->api->alloc_node(node.ctx, aligned, kAlignment,
                                              &allocation->buf);
    if (rc) return rc;

    allocation->node_id = node_id;
    allocation->dmabuf_fd = -1;
    return 0;
}

int MemoryPoolTransferEngine::Free(Allocation* allocation) {
    if (!allocation || !allocation->valid()) return 0;
    if (!IsOpen()) return -ENODEV;
    if (allocation->node_id >= context_->nodes.size()) return -EINVAL;
    if (allocation->mapped()) return -EBUSY;

    Node& node = context_->nodes[allocation->node_id];
    const int rc = context_->api->free(node.ctx, &allocation->buf);
    if (rc) return rc;

    if (allocation->dmabuf_fd >= 0) {
        close(allocation->dmabuf_fd);
        allocation->dmabuf_fd = -1;
    }
    allocation->buf = {};
    allocation->node_id = 0;
    return 0;
}

int MemoryPoolTransferEngine::TargetRange(const Allocation& allocation,
                                          uint64_t offset, size_t length,
                                          uint64_t* target_addr) const {
    if (!IsOpen() || !allocation.valid() || !target_addr) return -EINVAL;
    if (allocation.node_id >= context_->nodes.size()) return -EINVAL;
    return context_->api->target_range(&allocation.buf, offset, length,
                                       target_addr);
}

int MemoryPoolTransferEngine::ExportDmaBuf(Allocation* allocation, int flags,
                                            int* dmabuf_fd) const {
    if (!IsOpen() || !allocation || !allocation->valid() || !dmabuf_fd)
        return -EINVAL;
    if (allocation->node_id >= context_->nodes.size()) return -EINVAL;
    if (allocation->dmabuf_fd >= 0) return -EBUSY;

    int exported_fd = -1;
    const int rc = context_->api->export_dmabuf(
        context_->nodes[allocation->node_id].ctx, &allocation->buf, flags,
        &exported_fd);
    if (rc) return rc;

    const int owned_fd = dup(exported_fd);
    if (owned_fd < 0) {
        const int saved_errno = errno;
        close(exported_fd);
        return -saved_errno;
    }

    // Caller owns exported_fd; Allocation owns an independent duplicate.
    allocation->dmabuf_fd = owned_fd;
    *dmabuf_fd = exported_fd;
    return 0;
}

int MemoryPoolTransferEngine::Map(Allocation* allocation, size_t offset,
                                  size_t length, void** cpu_addr) {
    if (!IsOpen() || !allocation || !allocation->valid() || !cpu_addr)
        return -EINVAL;
    if (allocation->node_id >= context_->nodes.size()) return -EINVAL;
    if (allocation->mapped()) return -EBUSY;
    if (offset > allocation->buf.size || length > allocation->buf.size - offset)
        return -EINVAL;

    const int rc = context_->api->bo_map(
        context_->nodes[allocation->node_id].ctx, &allocation->buf, offset,
        length);
    if (rc) return rc;
    *cpu_addr = allocation->buf.cpu_addr;
    return 0;
}

int MemoryPoolTransferEngine::Unmap(Allocation* allocation, size_t length) {
    if (!IsOpen() || !allocation || !allocation->valid()) return -EINVAL;
    if (allocation->node_id >= context_->nodes.size()) return -EINVAL;
    if (!allocation->mapped()) return -EINVAL;
    if (length != allocation->buf.mapped_len) return -EINVAL;

    const int rc = context_->api->bo_unmap(&allocation->buf, length);
    if (rc) return rc;
    return 0;
}

}  // namespace mooncake

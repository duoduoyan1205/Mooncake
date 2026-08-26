#include "memory_pool_transfer_engine.h"

#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace mooncake {

namespace {

constexpr uint64_t kAlignment = 4096;

std::vector<std::string> SplitDevices(const std::string &devices)
{
    std::vector<std::string> result;
    std::stringstream stream(devices);
    std::string device;

    while (std::getline(stream, device, ',')) {
        const size_t begin = device.find_first_not_of(" \t\r\n");
        if (begin == std::string::npos) {
            continue;
        }

        const size_t end = device.find_last_not_of(" \t\r\n");
        result.push_back(device.substr(begin, end - begin + 1));
    }

    return result;
}

}  // namespace

MemoryPoolTransferEngine::Allocation::~Allocation() = default;

MemoryPoolTransferEngine::Allocation::Allocation(Allocation &&other) noexcept
    : node_id(other.node_id),
      buf(other.buf),
      dmabuf_fd(other.dmabuf_fd)
{
    other.node_id = 0;
    other.buf = {};
    other.dmabuf_fd = -1;
}

MemoryPoolTransferEngine::Allocation &
MemoryPoolTransferEngine::Allocation::operator=(Allocation &&other) noexcept
{
    if (this == &other) {
        return *this;
    }

    node_id = other.node_id;
    buf = other.buf;
    dmabuf_fd = other.dmabuf_fd;

    other.node_id = 0;
    other.buf = {};
    other.dmabuf_fd = -1;

    return *this;
}

struct MemoryPoolTransferEngine::Node {
    std::string device_path;
    amdgpu_mpu_ctx_t *ctx = nullptr;
    uint64_t capacity = 0;
    amdgpu_mpu_caps_t caps{};
};

struct MemoryPoolTransferEngine::Context {
    explicit Context(std::string library) : sueverbs_library(std::move(library)) {}

    std::string sueverbs_library;
    std::vector<Node> nodes;
    uint64_t next_node = 0;
    size_t active_allocations = 0;
    mutable std::mutex mutex;
};

MemoryPoolTransferEngine::MemoryPoolTransferEngine(
    std::string sueverbs_library,
    std::string device_paths)
    : context_(
          std::make_unique<Context>(std::move(sueverbs_library)))
{
    for (const std::string &device : SplitDevices(device_paths)) {
        Node node;
        node.device_path = device;
        context_->nodes.push_back(std::move(node));
    }
}

MemoryPoolTransferEngine::~MemoryPoolTransferEngine()
{
    Close();
}

int MemoryPoolTransferEngine::Open()
{
    if (!context_ || context_->nodes.empty()) {
        return -EINVAL;
    }

    if (IsOpen()) {
        return 0;
    }

    for (Node &node : context_->nodes) {
        const int open_ret =
            amdgpu_mpu_open(node.device_path.c_str(), &node.ctx);
        if (open_ret != 0 || node.ctx == nullptr) {
            Close();
            return open_ret != 0 ? open_ret : -ENODEV;
        }

        amdgpu_mpu_caps_t caps{};
        const int caps_ret = amdgpu_mpu_get_caps(node.ctx, &caps);
        if (caps_ret != 0 || caps.mem_size == 0) {
            amdgpu_mpu_close(node.ctx);
            node.ctx = nullptr;
            Close();
            return caps_ret != 0 ? caps_ret : -ENODEV;
        }

        node.capacity = caps.mem_size;
        node.caps = caps;
    }

    return 0;
}

void MemoryPoolTransferEngine::Close()
{
    if (!context_) {
        return;
    }

    std::lock_guard<std::mutex> lock(context_->mutex);

    if (context_->active_allocations != 0) {
        return;
    }

    for (Node &node : context_->nodes) {
        if (node.ctx != nullptr) {
            amdgpu_mpu_close(node.ctx);
        }

        node.ctx = nullptr;
        node.capacity = 0;
        node.caps = {};
    }

    context_->next_node = 0;
}

bool MemoryPoolTransferEngine::IsOpen() const
{
    if (!context_ || context_->nodes.empty()) {
        return false;
    }

    for (const Node &node : context_->nodes) {
        if (node.ctx == nullptr) {
            return false;
        }
    }

    return true;
}

size_t MemoryPoolTransferEngine::NodeCount() const
{
    return context_ ? context_->nodes.size() : 0;
}

uint64_t MemoryPoolTransferEngine::NodeCapacity(uint32_t node_id) const
{
    if (!context_ || node_id >= context_->nodes.size()) {
        return 0;
    }

    return context_->nodes[node_id].capacity;
}

uint64_t MemoryPoolTransferEngine::Capacity() const
{
    if (!context_) {
        return 0;
    }

    uint64_t total = 0;

    for (const Node &node : context_->nodes) {
        if (UINT64_MAX - total < node.capacity) {
            return UINT64_MAX;
        }

        total += node.capacity;
    }

    return total;
}

int MemoryPoolTransferEngine::Allocate(
    uint64_t size,
    Allocation *allocation)
{
    if (!IsOpen() || allocation == nullptr || size == 0) {
        return -EINVAL;
    }

    if (allocation->valid()) {
        return -EBUSY;
    }

    if (size > UINT64_MAX - (kAlignment - 1)) {
        return -EINVAL;
    }

    const uint64_t aligned_size =
        (size + kAlignment - 1) & ~(kAlignment - 1);

    std::lock_guard<std::mutex> lock(context_->mutex);

    const uint32_t node_id =
        context_->next_node++ % context_->nodes.size();

    const int ret = amdgpu_mpu_alloc_node(
        context_->nodes[node_id].ctx,
        aligned_size,
        kAlignment,
        &allocation->buf);
    if (ret != 0) {
        return ret;
    }

    allocation->node_id = node_id;
    ++context_->active_allocations;

    return 0;
}

int MemoryPoolTransferEngine::Free(Allocation *allocation)
{
    if (allocation == nullptr || !allocation->valid()) {
        return 0;
    }

    if (!IsOpen() || allocation->node_id >= context_->nodes.size()) {
        return -ENODEV;
    }

    if (allocation->mapped()) {
        return -EBUSY;
    }

    std::lock_guard<std::mutex> lock(context_->mutex);

    const int ret = amdgpu_mpu_free(
        context_->nodes[allocation->node_id].ctx,
        &allocation->buf);
    if (ret != 0) {
        return ret;
    }

    allocation->buf = {};
    allocation->node_id = 0;

    if (context_->active_allocations > 0) {
        --context_->active_allocations;
    }

    return 0;
}

int MemoryPoolTransferEngine::TargetRange(
    const Allocation &allocation,
    uint64_t offset,
    size_t length,
    uint64_t *target_address) const
{
    if (!IsOpen() || !allocation.valid() || target_address == nullptr ||
        allocation.node_id >= context_->nodes.size()) {
        return -EINVAL;
    }

    return amdgpu_mpu_target_range(
        &allocation.buf,
        offset,
        length,
        target_address);
}

int MemoryPoolTransferEngine::ExportDmaBuf(
    Allocation *allocation,
    int flags,
    int *dma_buf_fd) const
{
    if (!IsOpen() || allocation == nullptr || !allocation->valid() ||
        dma_buf_fd == nullptr || allocation->node_id >= context_->nodes.size()) {
        return -EINVAL;
    }

    return amdgpu_mpu_bo_export_dmabuf(
        context_->nodes[allocation->node_id].ctx,
        &allocation->buf,
        flags,
        dma_buf_fd);
}

int MemoryPoolTransferEngine::ImportDmaBuf(
    int fd,
    uint64_t address,
    uint64_t length,
    DmaBufType type,
    ImportedDmaBuf *imported) const
{
    if (!IsOpen() || imported == nullptr || fd < 0 || length == 0) {
        return -EINVAL;
    }

    if (imported->valid()) {
        return -EBUSY;
    }

    struct stat dma_buf_stat {};
    if (fstat(fd, &dma_buf_stat) != 0) {
        return -errno;
    }

    const int duplicated_fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (duplicated_fd < 0) {
        return -errno;
    }

    imported->fd = duplicated_fd;
    imported->length = length;
    imported->address = address;
    imported->type = type;

    return 0;
}

int MemoryPoolTransferEngine::ReleaseDmaBuf(ImportedDmaBuf *imported) const
{
    if (imported == nullptr || !imported->valid()) {
        return 0;
    }

    if (close(imported->fd) != 0) {
        return -errno;
    }

    *imported = {};
    return 0;
}

int MemoryPoolTransferEngine::SubmitDmaBufTransfer(
    int fd,
    uint64_t target_address,
    uint64_t source_offset,
    uint64_t length,
    uint32_t op,
    uint64_t *cookie)
{
    if (!IsOpen() || fd < 0 || target_address == 0 || length == 0 ||
        cookie == nullptr) {
        return -EINVAL;
    }

    for (Node &node : context_->nodes) {
        const uint64_t base = node.caps.mem_base;

        if (target_address < base) {
            continue;
        }

        const uint64_t offset = target_address - base;
        if (offset > node.caps.mem_size) {
            continue;
        }

        if (length > node.caps.mem_size - offset) {
            continue;
        }

        return amdgpu_mpu_sue_submit(
            node.ctx,
            fd,
            target_address,
            source_offset,
            length,
            op,
            0,
            cookie);
    }

    return -ERANGE;
}

int MemoryPoolTransferEngine::GetDmaBufTransferStatus(
    uint64_t cookie,
    amdgpu_mpu_sue_status_t *status)
{
    if (!IsOpen() || cookie == 0 || status == nullptr) {
        return -EINVAL;
    }

    for (Node &node : context_->nodes) {
        const int ret = amdgpu_mpu_sue_status(
            node.ctx,
            cookie,
            status);

        if (ret != -EAGAIN) {
            return ret;
        }
    }

    return -EAGAIN;
}

int MemoryPoolTransferEngine::Map(
    Allocation *allocation,
    size_t offset,
    size_t length,
    void **address)
{
    if (!IsOpen() || allocation == nullptr || !allocation->valid() ||
        address == nullptr || allocation->node_id >= context_->nodes.size() ||
        allocation->mapped() || offset > allocation->buf.size ||
        length > allocation->buf.size - offset) {
        return -EINVAL;
    }

    const int ret = amdgpu_mpu_bo_map(
        context_->nodes[allocation->node_id].ctx,
        &allocation->buf,
        offset,
        length);
    if (ret != 0) {
        return ret;
    }

    *address = allocation->buf.cpu_addr;
    return 0;
}

int MemoryPoolTransferEngine::Unmap(
    Allocation *allocation,
    size_t length)
{
    if (!IsOpen() || allocation == nullptr || !allocation->valid() ||
        allocation->node_id >= context_->nodes.size() ||
        !allocation->mapped() || length != allocation->buf.mapped_len) {
        return -EINVAL;
    }

    return amdgpu_mpu_bo_unmap(&allocation->buf, length);
}

}  // namespace mooncake

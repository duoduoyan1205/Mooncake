#include "transport/memory_pool_transport/memory_pool_transport.h"

#include <cerrno>
#include <cstring>
#include <memory>

#include "common.h"
#include "error.h"

namespace mooncake {

MemoryPoolTransport::MemoryPoolTransport() = default;
MemoryPoolTransport::~MemoryPoolTransport() = default;

int MemoryPoolTransport::install(
    std::string& local_server_name,
    std::shared_ptr<TransferMetadata> metadata,
    std::shared_ptr<Topology> topo) {
    (void)topo;

    metadata_ = std::move(metadata);
    local_server_name_ = local_server_name;

    const char* devices = std::getenv("MC_MPU_DEVICE_PATHS");
    if (!devices || !*devices) {
        LOG(ERROR) << "MemoryPoolTransport: MC_MPU_DEVICE_PATHS is not set";
        return -EINVAL;
    }

    const char* sueverbs = std::getenv("MC_SUEVERBS_LIBRARY");
    engine_ = std::make_unique<MemoryPoolTransferEngine>(
        sueverbs ? sueverbs : "", devices);
    return engine_->Open();
}

int MemoryPoolTransport::registerLocalMemory(
    void* addr, size_t length, const std::string& location,
    bool remote_accessible, bool update_metadata) {
    (void)location;
    (void)remote_accessible;

    if (!addr || !length || !engine_ || !engine_->IsOpen()) return -EINVAL;
    local_buffers_[addr] = length;

    if (!metadata_ || !update_metadata) return 0;

    TransferMetadata::BufferDesc desc;
    desc.name = local_server_name_;
    desc.addr = reinterpret_cast<uint64_t>(addr);
    desc.length = length;
#ifdef ENABLE_MULTI_PROTOCOL
    desc.protocol = "memory_pool";
#endif
    return metadata_->addLocalMemoryBuffer(desc, true);
}

int MemoryPoolTransport::unregisterLocalMemory(void* addr,
                                                bool update_metadata) {
    if (!addr) return -EINVAL;
    local_buffers_.erase(addr);
    if (metadata_ && update_metadata)
        return metadata_->removeLocalMemoryBuffer(addr, true);
    return 0;
}

int MemoryPoolTransport::registerLocalMemoryBatch(
    const std::vector<Transport::BufferEntry>& buffer_list,
    const std::string& location) {
    for (const auto& buffer : buffer_list) {
        int rc = registerLocalMemory(buffer.addr, buffer.length, location,
                                     true, false);
        if (rc) return rc;
    }
    return metadata_ ? metadata_->updateLocalSegmentDesc() : 0;
}

int MemoryPoolTransport::unregisterLocalMemoryBatch(
    const std::vector<void*>& addr_list) {
    int first_error = 0;
    for (void* addr : addr_list) {
        int rc = unregisterLocalMemory(addr, false);
        if (rc && !first_error) first_error = rc;
    }
    if (metadata_) {
        int rc = metadata_->updateLocalSegmentDesc();
        if (!first_error) first_error = rc;
    }
    return first_error;
}

Status MemoryPoolTransport::resolveTarget(const TransferRequest& request,
                                          uint64_t* target_addr,
                                          size_t* available) const {
    if (!metadata_ || !target_addr || !available)
        return Status::InvalidArgument("Invalid Memory Pool target arguments");

    auto segment = metadata_->getSegmentDescByID(request.target_id);
    if (!segment)
        return Status::InvalidArgument("Invalid target segment ID " +
                                       std::to_string(request.target_id));

    for (const auto& buffer : segment->buffers) {
        const uint64_t base = buffer.target_addr ? buffer.target_addr : buffer.addr;
        if (request.target_offset < base) continue;
        const uint64_t offset = request.target_offset - base;
        if (offset <= buffer.length &&
            request.length <= buffer.length - offset) {
            *target_addr = base + offset;
            *available = buffer.length - offset;
            return Status::OK();
        }
    }

    return Status::InvalidArgument(
        "Memory Pool target offset is outside registered buffers");
}

Status MemoryPoolTransport::submitTransfer(
    BatchID batch_id, const std::vector<TransferRequest>& entries) {
    auto& batch = Transport::toBatchDesc(batch_id);
    if (batch.task_list.size() + entries.size() > batch.batch_size)
        return Status::TooManyRequests("Exceed the limitation of batch capacity");

    const size_t first = batch.task_list.size();
    batch.task_list.resize(first + entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        auto& task = batch.task_list[first + i];
        task.batch_id = batch_id;
        task.transport_ = this;
        task.request = &entries[i];
        auto rc = submitTask(&task);
        if (!rc.ok()) return rc;
    }
    return Status::OK();
}

Status MemoryPoolTransport::submitTransferTask(
    const std::vector<TransferTask*>& task_list) {
    for (auto* task : task_list) {
        if (!task || !task->request)
            return Status::InvalidArgument("Invalid Memory Pool transfer task");
        auto rc = submitTask(task);
        if (!rc.ok()) return rc;
    }
    return Status::OK();
}

Status MemoryPoolTransport::submitTask(TransferTask* task) {
    auto& request = *task->request;
    task->total_bytes = request.length;

    uint64_t target_addr = 0;
    size_t available = 0;
    auto rc = resolveTarget(request, &target_addr, &available);
    if (!rc.ok()) return rc;
    if (request.length > available)
        return Status::InvalidArgument("Memory Pool transfer exceeds target buffer");

    auto* slice = getSliceCache().allocate();
    slice->source_addr = request.source;
    slice->length = request.length;
    slice->opcode = request.opcode;
    slice->target_id = request.target_id;
    slice->task = task;
    slice->status = Slice::PENDING;
    slice->tcp.dest_addr = target_addr;
    task->slice_list.push_back(slice);
    __sync_fetch_and_add(&task->slice_count, 1);

    // The current MPU ABI 1.8 exposes allocation/target-address/DMA-BUF/mmap,
    // but no CPU-side memcpy primitive. Until the GPU/NIC DMA-BUF importer is
    // connected, this transport intentionally uses the mapped target address
    // for the host-visible validation path only.
    if (!request.source || !target_addr) {
        slice->markFailed();
        return Status::InvalidArgument("Invalid Memory Pool transfer address");
    }

    void* target = reinterpret_cast<void*>(target_addr);
    int err = 0;
    if (request.opcode == TransferRequest::READ)
        std::memcpy(request.source, target, request.length);
    else
        std::memcpy(target, request.source, request.length);

    if (err) {
        slice->markFailed();
        return Status::InternalError("Memory Pool host transfer failed");
    }
    slice->markSuccess();
    return Status::OK();
}

Status MemoryPoolTransport::getTransferStatus(BatchID batch_id, size_t task_id,
                                               TransferStatus& status) {
    auto& batch = Transport::toBatchDesc(batch_id);
    if (task_id >= batch.task_list.size())
        return Status::InvalidArgument("Task ID out of range");

    auto& task = batch.task_list[task_id];
    status.transferred_bytes = task.transferred_bytes;
    if (task.success_slice_count + task.failed_slice_count == task.slice_count) {
        status.s = task.failed_slice_count ? TransferStatusEnum::FAILED
                                           : TransferStatusEnum::COMPLETED;
        task.is_finished = true;
    } else {
        status.s = TransferStatusEnum::WAITING;
    }
    return Status::OK();
}

}  // namespace mooncake

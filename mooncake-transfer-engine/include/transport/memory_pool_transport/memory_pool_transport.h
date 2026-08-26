#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "memory_pool_transfer_engine.h"
#include "transport/transport.h"

namespace mooncake {

class MemoryPoolTransport final : public Transport {
 public:
    MemoryPoolTransport();
    ~MemoryPoolTransport() override;

    Status submitTransfer(
        BatchID batch_id,
        const std::vector<TransferRequest>& entries) override;

    Status submitTransferTask(
        const std::vector<TransferTask*>& task_list) override;

    Status getTransferStatus(BatchID batch_id, size_t task_id,
                              TransferStatus& status) override;

 private:
    int install(std::string& local_server_name,
                std::shared_ptr<TransferMetadata> metadata,
                std::shared_ptr<Topology> topo) override;

    int registerLocalMemory(void* addr, size_t length,
                            const std::string& location,
                            bool remote_accessible,
                            bool update_metadata) override;

    int unregisterLocalMemory(void* addr,
                              bool update_metadata = true) override;

    int registerLocalMemoryBatch(
        const std::vector<Transport::BufferEntry>& buffer_list,
        const std::string& location) override;

    int unregisterLocalMemoryBatch(
        const std::vector<void*>& addr_list) override;

    const char* getName() const override { return "memory_pool"; }

    Status submitTask(TransferTask* task);
    Status resolveTarget(const TransferRequest& request, uint64_t* target_addr,
                         size_t* available) const;

    std::unique_ptr<MemoryPoolTransferEngine> engine_;
    std::unordered_map<void*, size_t> local_buffers_;
};

}  // namespace mooncake

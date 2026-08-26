#pragma once
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "memory_pool_transfer_engine.h"
#include "transport/transport.h"
namespace mooncake {
class MemoryPoolTransport final : public Transport {
 public:
    using DmaBufType=MemoryPoolTransferEngine::DmaBufType;
    using ImportedDmaBuf=MemoryPoolTransferEngine::ImportedDmaBuf;
    MemoryPoolTransport(); ~MemoryPoolTransport() override;
    Status submitTransfer(BatchID,const std::vector<TransferRequest>&) override;
    Status submitTransferTask(const std::vector<TransferTask*>&) override;
    Status getTransferStatus(BatchID,size_t,TransferStatus&) override;
    int importGpuDmaBuf(int dmabuf_fd,uint64_t device_addr,uint64_t length,ImportedDmaBuf* imported);
    int importNicDmaBuf(int dmabuf_fd,uint64_t device_addr,uint64_t length,ImportedDmaBuf* imported);
    int releaseDmaBuf(ImportedDmaBuf* imported);
    MemoryPoolTransferEngine* engine(){return engine_.get();}
 private:
    int install(std::string&,std::shared_ptr<TransferMetadata>,std::shared_ptr<Topology>) override;
    int registerLocalMemory(void*,size_t,const std::string&,bool,bool) override;
    int unregisterLocalMemory(void*,bool=true) override;
    int registerLocalMemoryBatch(const std::vector<Transport::BufferEntry>&,const std::string&) override;
    int unregisterLocalMemoryBatch(const std::vector<void*>&) override;
    const char* getName() const override{return "memory_pool";}
    Status submitTask(TransferTask*);
    Status resolveTarget(const TransferRequest&,uint64_t*,size_t*) const;
    bool resolveSource(void*,size_t,int*,uint64_t*) const;
    std::unique_ptr<MemoryPoolTransferEngine> engine_;
    std::unordered_map<void*,size_t> local_buffers_;
    std::unordered_map<uint64_t,ImportedDmaBuf> dma_bufs_;
    std::unordered_map<TransferTask*,uint64_t> cookies_;
    mutable std::mutex dma_buf_mutex_;
};
} // namespace mooncake

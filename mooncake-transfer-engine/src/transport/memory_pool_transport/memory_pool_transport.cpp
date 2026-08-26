#include "transport/memory_pool_transport/memory_pool_transport.h"
#include <cerrno>
#include <cstdlib>
#include <memory>
#include <limits>
#include "common.h"
#include "error.h"
namespace mooncake {
MemoryPoolTransport::MemoryPoolTransport()=default;
MemoryPoolTransport::~MemoryPoolTransport(){
    std::lock_guard<std::mutex> lock(dma_buf_mutex_);
    if(engine_){for(auto &p:dma_bufs_)engine_->ReleaseDmaBuf(&p.second);}
}
int MemoryPoolTransport::install(std::string&name,std::shared_ptr<TransferMetadata>md,std::shared_ptr<Topology>topo){
    (void)topo;metadata_=std::move(md);local_server_name_=name;
    const char*d=std::getenv("MC_MPU_DEVICE_PATHS");if(!d||!*d){LOG(ERROR)<<"MC_MPU_DEVICE_PATHS is not set";return-EINVAL;}
    const char*s=std::getenv("MC_SUEVERBS_LIBRARY");engine_=std::make_unique<MemoryPoolTransferEngine>(s?s:"",d);return engine_->Open();
}
int MemoryPoolTransport::registerLocalMemory(void*addr,size_t len,const std::string&loc,bool remote,bool update){
    (void)loc;(void)remote;if(!addr||!len||!engine_||!engine_->IsOpen())return-EINVAL;local_buffers_[addr]=len;
    if(!metadata_||!update)return 0;TransferMetadata::BufferDesc desc;desc.name=local_server_name_;desc.addr=(uint64_t)addr;desc.length=len;
#ifdef ENABLE_MULTI_PROTOCOL
    desc.protocol="memory_pool";
#endif
    return metadata_->addLocalMemoryBuffer(desc,true);
}
int MemoryPoolTransport::unregisterLocalMemory(void*addr,bool update){if(!addr)return-EINVAL;local_buffers_.erase(addr);if(metadata_&&update)return metadata_->removeLocalMemoryBuffer(addr,true);return 0;}
int MemoryPoolTransport::registerLocalMemoryBatch(const std::vector<Transport::BufferEntry>&list,const std::string&loc){for(const auto&b:list){int r=registerLocalMemory(b.addr,b.length,loc,true,false);if(r)return r;}return metadata_?metadata_->updateLocalSegmentDesc():0;}
int MemoryPoolTransport::unregisterLocalMemoryBatch(const std::vector<void*>&list){int e=0;for(void*a:list){int r=unregisterLocalMemory(a,false);if(r&&!e)e=r;}if(metadata_){int r=metadata_->updateLocalSegmentDesc();if(!e)e=r;}return e;}
int MemoryPoolTransport::importGpuDmaBuf(int fd,uint64_t addr,uint64_t len,ImportedDmaBuf*out){if(!engine_)return-ENODEV;int r=engine_->ImportDmaBuf(fd,addr,len,DmaBufType::GPU,out);if(!r){std::lock_guard<std::mutex>l(dma_buf_mutex_);dma_bufs_[addr]=*out;}return r;}
int MemoryPoolTransport::importNicDmaBuf(int fd,uint64_t addr,uint64_t len,ImportedDmaBuf*out){if(!engine_)return-ENODEV;int r=engine_->ImportDmaBuf(fd,addr,len,DmaBufType::NIC,out);if(!r){std::lock_guard<std::mutex>l(dma_buf_mutex_);dma_bufs_[addr]=*out;}return r;}
int MemoryPoolTransport::releaseDmaBuf(ImportedDmaBuf*out){if(!engine_||!out)return-ENODEV;std::lock_guard<std::mutex>l(dma_buf_mutex_);int r=engine_->ReleaseDmaBuf(out);if(!r){for(auto it=dma_bufs_.begin();it!=dma_bufs_.end();++it)if(it->second.fd==out->fd){dma_bufs_.erase(it);break;}}return r;}
bool MemoryPoolTransport::resolveSource(void*source,size_t len,int*fd,uint64_t*off)const{uint64_t a=(uint64_t)source;std::lock_guard<std::mutex>l(dma_buf_mutex_);for(const auto&p:dma_bufs_){if(a>=p.second.address&&a-p.second.address<=p.second.length&&len<=p.second.length-(a-p.second.address)){*fd=p.second.fd;*off=a-p.second.address;return true;}}return false;}
Status MemoryPoolTransport::resolveTarget(const TransferRequest&r,uint64_t*addr,size_t*avail)const{
    if(!metadata_||!addr||!avail)return Status::InvalidArgument("Invalid Memory Pool target arguments");auto seg=metadata_->getSegmentDescByID(r.target_id);if(!seg)return Status::InvalidArgument("Invalid target segment ID "+std::to_string(r.target_id));
    for(const auto&b:seg->buffers){uint64_t base=b.target_addr?b.target_addr:b.addr;if(r.target_offset<base)continue;uint64_t off=r.target_offset-base;if(off<=b.length&&r.length<=b.length-off){*addr=base+off;*avail=b.length-off;return Status::OK();}}
    return Status::InvalidArgument("Memory Pool target offset is outside registered buffers");
}
Status MemoryPoolTransport::submitTransfer(BatchID id,const std::vector<TransferRequest>&e){auto&b=Transport::toBatchDesc(id);if(b.task_list.size()+e.size()>b.batch_size)return Status::TooManyRequests("Exceed batch capacity");size_t n=b.task_list.size();b.task_list.resize(n+e.size());for(size_t i=0;i<e.size();++i){auto&t=b.task_list[n+i];t.batch_id=id;t.transport_=this;t.request=&e[i];auto r=submitTask(&t);if(!r.ok())return r;}return Status::OK();}
Status MemoryPoolTransport::submitTransferTask(const std::vector<TransferTask*>&list){for(auto*t:list){if(!t||!t->request)return Status::InvalidArgument("Invalid Memory Pool transfer task");auto r=submitTask(t);if(!r.ok())return r;}return Status::OK();}
Status MemoryPoolTransport::submitTask(TransferTask*t){
    auto&r=*t->request;t->total_bytes=r.length;if(!r.length)return Status::InvalidArgument("Zero length transfer");
    int fd=-1;uint64_t src_off=0;if(!resolveSource(r.source,r.length,&fd,&src_off))return Status::InvalidArgument("Source is not a registered GPU/NIC DMA-BUF");
    uint64_t target=0;size_t avail=0;auto rc=resolveTarget(r,&target,&avail);if(!rc.ok())return rc;if(r.length>avail)return Status::InvalidArgument("Transfer exceeds target buffer");
    auto*slice=getSliceCache().allocate();slice->source_addr=r.source;slice->length=r.length;slice->opcode=r.opcode;slice->target_id=r.target_id;slice->task=t;slice->status=Slice::POSTED;slice->tcp.dest_addr=target;t->slice_list.push_back(slice);__sync_fetch_and_add(&t->slice_count,1);
    uint64_t cookie=0;uint32_t op=(r.opcode==TransferRequest::READ)?AMDGPU_MPU_SUE_OP_READ:AMDGPU_MPU_SUE_OP_WRITE;
    int ret=engine_->SubmitDmaBufTransfer(fd,target,r.target_offset,src_off,r.length,op,&cookie);
    if(ret){slice->markFailed();return Status::InvalidArgument("MPU SUE submit failed: "+std::to_string(ret));}
    {std::lock_guard<std::mutex>l(dma_buf_mutex_);cookies_[t]=cookie;}
    return Status::OK();
}
Status MemoryPoolTransport::getTransferStatus(BatchID id,size_t tid,TransferStatus&status){
    auto&b=Transport::toBatchDesc(id);if(tid>=b.task_list.size())return Status::InvalidArgument("Task ID out of range");auto&t=b.task_list[tid];status.transferred_bytes=t.transferred_bytes;
    uint64_t cookie=0;{std::lock_guard<std::mutex>l(dma_buf_mutex_);auto it=cookies_.find(&t);if(it==cookies_.end()){status.s=t.failed_slice_count?FAILED:(t.success_slice_count?COMPLETED:WAITING);return Status::OK();}cookie=it->second;}
    amdgpu_mpu_sue_status_t s{};int r=engine_->GetDmaBufTransferStatus(cookie,&s);if(r==-EAGAIN){status.s=WAITING;return Status::OK();}if(r){status.s=FAILED;return Status::OK();}
    if(s.status==0){for(auto*sl:t.slice_list)if(sl->status==Slice::POSTED){sl->markSuccess();break;}status.transferred_bytes=t.transferred_bytes;status.s=COMPLETED;}else{for(auto*sl:t.slice_list)if(sl->status==Slice::POSTED){sl->markFailed();break;}status.s=FAILED;}
    std::lock_guard<std::mutex>l(dma_buf_mutex_);cookies_.erase(&t);return Status::OK();
}
} // namespace mooncake

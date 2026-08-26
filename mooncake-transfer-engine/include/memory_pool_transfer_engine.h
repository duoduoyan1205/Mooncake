#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <libamdgpu_mpu.h>
namespace mooncake {
class MemoryPoolTransferEngine {
 public:
    enum class DmaBufType:uint8_t{GPU=0,NIC=1};
    struct ImportedDmaBuf{int fd=-1;uint64_t length=0,address=0;DmaBufType type=DmaBufType::GPU;bool valid()const{return fd>=0&&length!=0;}};
    struct Allocation{uint32_t node_id=0;amdgpu_mpu_buf_t buf{};int dmabuf_fd=-1;Allocation()=default;~Allocation();Allocation(Allocation&&)noexcept;Allocation&operator=(Allocation&&)noexcept;Allocation(const Allocation&)=delete;Allocation&operator=(const Allocation&)=delete;bool valid()const{return buf.handle!=0;}bool mapped()const{return buf.cpu_addr!=nullptr;}};
    MemoryPoolTransferEngine(std::string,std::string);~MemoryPoolTransferEngine();MemoryPoolTransferEngine(const MemoryPoolTransferEngine&)=delete;MemoryPoolTransferEngine&operator=(const MemoryPoolTransferEngine&)=delete;
    int Open();void Close();bool IsOpen()const;size_t NodeCount()const;uint64_t Capacity()const;uint64_t NodeCapacity(uint32_t)const;int Allocate(uint64_t,Allocation*);int Free(Allocation*);int TargetRange(const Allocation&,uint64_t,size_t,uint64_t*)const;int ExportDmaBuf(Allocation*,int,int*)const;int ImportDmaBuf(int,uint64_t,uint64_t,DmaBufType,ImportedDmaBuf*);int ReleaseDmaBuf(ImportedDmaBuf*);int SubmitDmaBufTransfer(int,uint64_t,uint64_t,uint64_t,uint32_t,uint64_t*);int GetDmaBufTransferStatus(uint64_t,amdgpu_mpu_sue_status_t*);int Map(Allocation*,size_t,size_t,void**);int Unmap(Allocation*,size_t);
 private:struct Node;struct Context;std::unique_ptr<Context> context_;
};
} // namespace mooncake

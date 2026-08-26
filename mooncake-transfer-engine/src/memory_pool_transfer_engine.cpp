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
namespace mooncake {namespace {constexpr uint64_t kAlignment=4096;std::vector<std::string>SplitDevices(const std::string&s){std::vector<std::string>r;std::stringstream x(s);std::string i;while(std::getline(x,i,',')){auto a=i.find_first_not_of(" \t\r\n");if(a==std::string::npos)continue;auto b=i.find_last_not_of(" \t\r\n");r.push_back(i.substr(a,b-a+1));}return r;}}
MemoryPoolTransferEngine::Allocation::~Allocation()=default;
MemoryPoolTransferEngine::Allocation::Allocation(Allocation&&o)noexcept:node_id(o.node_id),buf(o.buf),dmabuf_fd(o.dmabuf_fd){o.node_id=0;o.buf={};o.dmabuf_fd=-1;}
MemoryPoolTransferEngine::Allocation&MemoryPoolTransferEngine::Allocation::operator=(Allocation&&o)noexcept{if(this==&o)return*this;node_id=o.node_id;buf=o.buf;dmabuf_fd=o.dmabuf_fd;o.node_id=0;o.buf={};o.dmabuf_fd=-1;return*this;}
struct MemoryPoolTransferEngine::Node{std::string device_path;amdgpu_mpu_ctx_t*ctx=nullptr;uint64_t capacity=0;amdgpu_mpu_caps_t caps{};};
struct MemoryPoolTransferEngine::Context{explicit Context(std::string){}std::vector<Node>nodes;uint64_t next_node=0;size_t active_allocations=0;mutable std::mutex mutex;};
MemoryPoolTransferEngine::MemoryPoolTransferEngine(std::string lib,std::string paths):context_(std::make_unique<Context>(std::move(lib))){for(const auto&d:SplitDevices(paths)){Node n;n.device_path=d;context_->nodes.push_back(std::move(n));}}
MemoryPoolTransferEngine::~MemoryPoolTransferEngine(){Close();}
int MemoryPoolTransferEngine::Open(){if(!context_||context_->nodes.empty())return-EINVAL;if(IsOpen())return 0;for(auto&n:context_->nodes){int r=amdgpu_mpu_open(n.device_path.c_str(),&n.ctx);if(r||!n.ctx){Close();return r?r:-ENODEV;}amdgpu_mpu_caps_t c{};r=amdgpu_mpu_get_caps(n.ctx,&c);if(r||!c.mem_size){amdgpu_mpu_close(n.ctx);n.ctx=nullptr;Close();return r?r:-ENODEV;}n.capacity=c.mem_size;n.caps=c;}return 0;}
void MemoryPoolTransferEngine::Close(){if(!context_)return;std::lock_guard<std::mutex>l(context_->mutex);if(context_->active_allocations)return;for(auto&n:context_->nodes){if(n.ctx)amdgpu_mpu_close(n.ctx);n.ctx=nullptr;n.capacity=0;n.caps={};}context_->next_node=0;}
bool MemoryPoolTransferEngine::IsOpen()const{if(!context_||context_->nodes.empty())return false;for(const auto&n:context_->nodes)if(!n.ctx)return false;return true;}size_t MemoryPoolTransferEngine::NodeCount()const{return context_?context_->nodes.size():0;}uint64_t MemoryPoolTransferEngine::NodeCapacity(uint32_t id)const{return context_&&id<context_->nodes.size()?context_->nodes[id].capacity:0;}uint64_t MemoryPoolTransferEngine::Capacity()const{uint64_t t=0;if(!context_)return 0;for(const auto&n:context_->nodes){if(UINT64_MAX-t<n.capacity)return UINT64_MAX;t+=n.capacity;}return t;}
int MemoryPoolTransferEngine::Allocate(uint64_t s,Allocation*a){if(!IsOpen()||!a||!s)return-EINVAL;if(a->valid())return-EBUSY;if(s>UINT64_MAX-(kAlignment-1))return-EINVAL;uint64_t z=(s+kAlignment-1)&~(kAlignment-1);std::lock_guard<std::mutex>l(context_->mutex);uint32_t id=context_->next_node++%context_->nodes.size();int r=amdgpu_mpu_alloc_node(context_->nodes[id].ctx,z,kAlignment,&a->buf);if(r)return r;a->node_id=id;++context_->active_allocations;return 0;}
int MemoryPoolTransferEngine::Free(Allocation*a){if(!a||!a->valid())return 0;if(!IsOpen()||a->node_id>=context_->nodes.size())return-ENODEV;if(a->mapped())return-EBUSY;std::lock_guard<std::mutex>l(context_->mutex);int r=amdgpu_mpu_free(context_->nodes[a->node_id].ctx,&a->buf);if(r)return r;a->buf={};a->node_id=0;if(context_->active_allocations)--context_->active_allocations;return 0;}
int MemoryPoolTransferEngine::TargetRange(const Allocation&a,uint64_t o,size_t l,uint64_t*out)const{if(!IsOpen()||!a.valid()||!out||a.node_id>=context_->nodes.size())return-EINVAL;return amdgpu_mpu_target_range(&a.buf,o,l,out);}
int MemoryPoolTransferEngine::ExportDmaBuf(Allocation*a,int flags,int*out)const{if(!IsOpen()||!a||!a->valid()||!out||a->node_id>=context_->nodes.size())return-EINVAL;return amdgpu_mpu_bo_export_dmabuf(context_->nodes[a->node_id].ctx,&a->buf,flags,out);}
int MemoryPoolTransferEngine::ImportDmaBuf(int fd,uint64_t addr,uint64_t len,DmaBufType type,ImportedDmaBuf*out){if(!IsOpen()||!out||fd<0||!len)return-EINVAL;if(out->valid())return-EBUSY;struct stat st{};if(fstat(fd,&st))return-errno;int d=fcntl(fd,F_DUPFD_CLOEXEC,0);if(d<0)return-errno;out->fd=d;out->length=len;out->address=addr;out->type=type;return 0;}
int MemoryPoolTransferEngine::ReleaseDmaBuf(ImportedDmaBuf*out){if(!out||!out->valid())return 0;if(close(out->fd))return-errno;*out={};return 0;}
int MemoryPoolTransferEngine::SubmitDmaBufTransfer(int fd,uint64_t target,uint64_t src_off,uint64_t len,uint32_t op,uint64_t*cookie){if(!IsOpen()||fd<0||!target||!len||!cookie)return-EINVAL;for(auto&n:context_->nodes){uint64_t base=n.caps.mem_base;if(target>=base&&target-base<=n.caps.mem_size&&len<=n.caps.mem_size-(target-base))return amdgpu_mpu_sue_submit(n.ctx,fd,target,src_off,len,op,0,cookie);}return-ERANGE;}
int MemoryPoolTransferEngine::GetDmaBufTransferStatus(uint64_t cookie,amdgpu_mpu_sue_status_t*s){if(!IsOpen()||!cookie||!s)return-EINVAL;for(auto&n:context_->nodes){int r=amdgpu_mpu_sue_status(n.ctx,cookie,s);if(r!=-EAGAIN)return r;}return-EAGAIN;}
int MemoryPoolTransferEngine::Map(Allocation*a,size_t o,size_t l,void**p){if(!IsOpen()||!a||!a->valid()||!p||a->node_id>=context_->nodes.size()||a->mapped()||o>a->buf.size||l>a->buf.size-o)return-EINVAL;int r=amdgpu_mpu_bo_map(context_->nodes[a->node_id].ctx,&a->buf,o,l);if(r)return r;*p=a->buf.cpu_addr;return 0;}
int MemoryPoolTransferEngine::Unmap(Allocation*a,size_t l){if(!IsOpen()||!a||!a->valid()||a->node_id>=context_->nodes.size()||!a->mapped()||l!=a->buf.mapped_len)return-EINVAL;return amdgpu_mpu_bo_unmap(&a->buf,l);}
} // namespace mooncake

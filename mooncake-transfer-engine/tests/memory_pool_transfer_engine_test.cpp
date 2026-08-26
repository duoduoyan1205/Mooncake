// Copyright 2026 Mooncake Authors
// Licensed under the Apache License, Version 2.0
#include <gtest/gtest.h>
#include <cstdlib>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>
#include "memory_pool_transfer_engine.h"
using namespace mooncake;
namespace {
constexpr size_t kTransferSize=2*1024*1024;
std::string EnvOrDefault(const char*n,const char*f){const char*v=getenv(n);return v&&*v?std::string(v):std::string(f);}
std::vector<std::string>SplitDevices(const std::string&d){std::vector<std::string>r;std::stringstream s(d);std::string x;while(std::getline(s,x,',')){auto a=x.find_first_not_of(" \t\r\n");if(a==std::string::npos)continue;auto b=x.find_last_not_of(" \t\r\n");r.push_back(x.substr(a,b-a+1));}return r;}
MemoryPoolTransferEngine OpenEngine(){return MemoryPoolTransferEngine(EnvOrDefault("MOONCAKE_SUEVERBS_LIBRARY","libsueverbs.so"),EnvOrDefault("MOONCAKE_MEMORY_POOL_DEVICES",EnvOrDefault("MOONCAKE_MEMORY_POOL_DEVICE","/dev/amdgpu-mpu0")));}
void RequireEngineOpen(MemoryPoolTransferEngine&e){if(!e.NodeCount())GTEST_SKIP()<<"No MPU devices configured";for(const auto&d:SplitDevices(EnvOrDefault("MOONCAKE_MEMORY_POOL_DEVICES",EnvOrDefault("MOONCAKE_MEMORY_POOL_DEVICE","/dev/amdgpu-mpu0"))))if(access(d.c_str(),R_OK|W_OK))GTEST_SKIP()<<"MPU device unavailable: "<<d;int r=e.Open();if(r)GTEST_SKIP()<<"Unable to open MPU backend, rc="<<r;ASSERT_TRUE(e.IsOpen());}
}
TEST(MemoryPoolTransferEngineTest,MultiNodeAllocationAndTargetRange){auto e=OpenEngine();RequireEngineOpen(e);ASSERT_GT(e.Capacity(),0u);std::vector<MemoryPoolTransferEngine::Allocation>a;a.reserve(e.NodeCount()+1);for(size_t i=0;i<e.NodeCount()+1;++i){MemoryPoolTransferEngine::Allocation x;ASSERT_EQ(e.Allocate(kTransferSize,&x),0);ASSERT_TRUE(x.valid());uint64_t addr=0;ASSERT_EQ(e.TargetRange(x,0,kTransferSize,&addr),0);ASSERT_NE(addr,0u);a.push_back(std::move(x));}for(auto&x:a)ASSERT_EQ(e.Free(&x),0);}
TEST(MemoryPoolTransferEngineTest,DmaBufExportAndImport){auto e=OpenEngine();RequireEngineOpen(e);MemoryPoolTransferEngine::Allocation p;ASSERT_EQ(e.Allocate(kTransferSize,&p),0);int fd=-1;ASSERT_EQ(e.ExportDmaBuf(&p,O_CLOEXEC,&fd),0);MemoryPoolTransferEngine::ImportedDmaBuf g,n;ASSERT_EQ(e.ImportDmaBuf(fd,0x10000000,kTransferSize,MemoryPoolTransferEngine::DmaBufType::GPU,&g),0);ASSERT_EQ(e.ImportDmaBuf(fd,0x20000000,kTransferSize,MemoryPoolTransferEngine::DmaBufType::NIC,&n),0);ASSERT_NE(g.fd,n.fd);ASSERT_EQ(e.ReleaseDmaBuf(&g),0);ASSERT_EQ(e.ReleaseDmaBuf(&n),0);close(fd);ASSERT_EQ(e.Free(&p),0);}
TEST(MemoryPoolTransferEngineTest,SueQueueSubmission){auto e=OpenEngine();RequireEngineOpen(e);MemoryPoolTransferEngine::Allocation src,dst;ASSERT_EQ(e.Allocate(kTransferSize,&src),0);ASSERT_EQ(e.Allocate(kTransferSize,&dst),0);int fd=-1;ASSERT_EQ(e.ExportDmaBuf(&src,O_CLOEXEC,&fd),0);uint64_t target=0;ASSERT_EQ(e.TargetRange(dst,0,kTransferSize,&target),0);uint64_t cookie=0;int r=e.SubmitDmaBufTransfer(fd,target,0,kTransferSize,AMDGPU_MPU_SUE_OP_READ,&cookie);ASSERT_EQ(r,0);ASSERT_NE(cookie,0u);amdgpu_mpu_sue_status_t st{};r=e.GetDmaBufTransferStatus(cookie,&st);ASSERT_TRUE(r==0||r==-11);close(fd);ASSERT_EQ(e.Free(&src),0);ASSERT_EQ(e.Free(&dst),0);}
TEST(MemoryPoolTransferEngineTest,ExternalDmaBufImport){const char*g=getenv("MOONCAKE_GPU_DMABUF_FD");const char*n=getenv("MOONCAKE_NIC_DMABUF_FD");if(!g&&!n)GTEST_SKIP()<<"Set MOONCAKE_GPU_DMABUF_FD and/or MOONCAKE_NIC_DMABUF_FD";auto e=OpenEngine();RequireEngineOpen(e);std::vector<MemoryPoolTransferEngine::ImportedDmaBuf>v;auto f=[&](const char*x,uint64_t a,MemoryPoolTransferEngine::DmaBufType t){if(!x)return;char*z=nullptr;long fd=strtol(x,&z,10);ASSERT_TRUE(z!=x&&*z=='\0');MemoryPoolTransferEngine::ImportedDmaBuf i;ASSERT_EQ(e.ImportDmaBuf((int)fd,a,kTransferSize,t,&i),0);v.push_back(std::move(i));};f(g,0x30000000,MemoryPoolTransferEngine::DmaBufType::GPU);f(n,0x40000000,MemoryPoolTransferEngine::DmaBufType::NIC);for(auto&i:v)ASSERT_EQ(e.ReleaseDmaBuf(&i),0);}

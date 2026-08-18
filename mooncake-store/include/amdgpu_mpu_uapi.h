#pragma once

#include <cstdint>
#include <linux/ioctl.h>

namespace mooncake::amdgpu_mpu {

constexpr unsigned long kIoctlBase = 'M';

constexpr uint32_t kPathPToD = 0;
constexpr uint32_t kPathDToP = 1;
constexpr uint32_t kPathDToPoolWrite = 2;
constexpr uint32_t kPathDToPoolRead = 3;

constexpr uint32_t kXferSignal = 1U << 0;
constexpr uint32_t kXferFence = 1U << 1;
constexpr uint32_t kXferNonblock = 1U << 2;
constexpr uint32_t kXferOrdered = 1U << 3;

struct IoctlAlloc { uint64_t size; uint64_t alignment; uint64_t handle; uint64_t global_addr; };
struct IoctlFree { uint64_t handle; };
struct IoctlXfer {
    uint64_t src_addr; uint64_t dst_addr; uint64_t length;
    uint32_t path; uint32_t flags; uint64_t cookie;
    int32_t fence_fd; uint32_t reserved; uint64_t src_handle; uint64_t dst_handle;
};
struct IoctlWait { uint64_t cookie; uint64_t timeout_ns; int32_t status; uint32_t reserved; };
struct IoctlMmap { uint64_t handle; uint64_t offset; uint64_t size; uint32_t flags; uint32_t reserved; };
struct IoctlCaps { uint32_t sue_version; uint32_t sue_caps; uint32_t num_queues; uint32_t flags; uint64_t mem_base; uint64_t mem_size; };

#define MOONCAKE_AMDGPU_MPU_IOCTL_ALLOC _IOWR(kIoctlBase, 0x00, mooncake::amdgpu_mpu::IoctlAlloc)
#define MOONCAKE_AMDGPU_MPU_IOCTL_FREE _IOW(kIoctlBase, 0x01, mooncake::amdgpu_mpu::IoctlFree)
#define MOONCAKE_AMDGPU_MPU_IOCTL_XFER _IOWR(kIoctlBase, 0x02, mooncake::amdgpu_mpu::IoctlXfer)
#define MOONCAKE_AMDGPU_MPU_IOCTL_WAIT _IOWR(kIoctlBase, 0x03, mooncake::amdgpu_mpu::IoctlWait)
#define MOONCAKE_AMDGPU_MPU_IOCTL_MMAP _IOWR(kIoctlBase, 0x05, mooncake::amdgpu_mpu::IoctlMmap)
#define MOONCAKE_AMDGPU_MPU_IOCTL_CAPS _IOR(kIoctlBase, 0x06, mooncake::amdgpu_mpu::IoctlCaps)

}  // namespace mooncake::amdgpu_mpu

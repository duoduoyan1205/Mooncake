#pragma once

#include <cstdint>
#include <linux/ioctl.h>

namespace mooncake::amdgpu_mpu {

// Userspace mirror of kernel/drm/amdgpu_mpu/amdgpu_mpu_uapi.h.
// Keep the layouts and ioctl numbers byte-for-byte compatible with the MPU
// DRM UAPI. The MPU is a DRM render device; it no longer exposes the old
// private ALLOC/FREE/XFER/WAIT ioctl family.
constexpr uint32_t kUapiVersion = 5;
constexpr unsigned long kDrmIoctlBase = 'd';
constexpr unsigned long kDrmCommandBase = 0x40;

struct GemCreate {
    uint64_t size;
    uint64_t alignment;
    uint32_t handle;
    uint32_t flags;
    uint64_t gpu_addr;
    uint64_t target_addr;
    uint64_t mmap_offset;
};

constexpr uint32_t kGemCreateNodeMemory = 1U << 0;

struct GemMmap {
    uint32_t handle;
    uint32_t flags;
    uint64_t offset;
};

struct VmMap {
    uint32_t handle;
    uint32_t flags;
    uint64_t gpu_addr;
};

struct VmUnmap {
    uint32_t handle;
    uint32_t reserved;
};

struct Caps {
    uint32_t version;
    uint32_t flags;
    uint32_t vmid;
    uint32_t page_shift;
    uint64_t va_start;
    uint64_t va_end;
    uint64_t mem_base;
    uint64_t mem_size;
};

constexpr uint32_t kCapMmu = 1U << 0;
constexpr uint32_t kCapGpuVm = 1U << 1;
constexpr uint32_t kCapSue = 1U << 3;
constexpr uint32_t kCapP2p = 1U << 4;

#define MOONCAKE_AMDGPU_MPU_GEM_CREATE \
    _IOWR(kDrmIoctlBase, kDrmCommandBase + 0x00, \
          mooncake::amdgpu_mpu::GemCreate)
#define MOONCAKE_AMDGPU_MPU_GEM_MMAP \
    _IOWR(kDrmIoctlBase, kDrmCommandBase + 0x01, \
          mooncake::amdgpu_mpu::GemMmap)
#define MOONCAKE_AMDGPU_MPU_VM_MAP \
    _IOWR(kDrmIoctlBase, kDrmCommandBase + 0x02, \
          mooncake::amdgpu_mpu::VmMap)
#define MOONCAKE_AMDGPU_MPU_VM_UNMAP \
    _IOW(kDrmIoctlBase, kDrmCommandBase + 0x03, \
         mooncake::amdgpu_mpu::VmUnmap)
#define MOONCAKE_AMDGPU_MPU_GET_CAPS \
    _IOR(kDrmIoctlBase, kDrmCommandBase + 0x04, \
         mooncake::amdgpu_mpu::Caps)

struct GemClose {
    uint32_t handle;
    uint32_t pad;
};

#define MOONCAKE_AMDGPU_MPU_GEM_CLOSE \
    _IOW(kDrmIoctlBase, 0x09, mooncake::amdgpu_mpu::GemClose)

}  // namespace mooncake::amdgpu_mpu

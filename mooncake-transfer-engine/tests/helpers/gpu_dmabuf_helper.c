/* SPDX-License-Identifier: MIT
 *
 * Real AMDGPU DMA-BUF importer and SDMA exerciser.
 *
 * ABI:
 *   gpu_dmabuf_helper <write|read> <size> <dma-buf-fd> <pattern>
 *
 * write: CPU staging buffer -> AMDGPU SDMA -> imported MPU BO
 * read:  imported MPU BO -> AMDGPU SDMA -> CPU staging buffer -> verify
 *
 * The SDMA packet below is the classic linear-copy packet used by AMDGPU
 * generations which expose the 7-DW COPY_LINEAR format.  This helper is a
 * hardware test, not an MPU-side copy engine.
 */
#include <amdgpu.h>
#include <amdgpu_drm.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SDMA_OPCODE_COPY 1
#define SDMA_COPY_SUB_OPCODE_LINEAR 0
#define SDMA_PACKET(op, sub_op, e) ((((e) & 0xffff) << 16) | (((sub_op) & 0xff) << 8) | ((op) & 0xff))
#define SDMA_COPY_DWORDS 7
#define SDMA_MAX_COPY (1ULL << 20)
#define AMDGPU_TIMEOUT_NS 10000000000ULL
#define PAGE_ALIGN 4096ULL

static int alloc_bo_and_map(amdgpu_device_handle dev, uint64_t size,
                            amdgpu_bo_handle *bo, void **cpu,
                            uint64_t *va, amdgpu_va_handle *va_handle)
{
    struct amdgpu_bo_alloc_request req = {0};
    int r;

    req.alloc_size = size;
    req.phys_alignment = PAGE_ALIGN;
    req.preferred_heap = AMDGPU_GEM_DOMAIN_GTT;
    req.flags = 0;

    r = amdgpu_bo_alloc(dev, &req, bo);
    if (r)
        return r;

    r = amdgpu_va_range_alloc(dev, amdgpu_gpu_va_range_general,
                              size, PAGE_ALIGN, 0, va, va_handle, 0);
    if (r)
        goto err_bo;

    r = amdgpu_bo_va_op(*bo, 0, size, *va, 0, AMDGPU_VA_OP_MAP);
    if (r)
        goto err_va;

    r = amdgpu_bo_cpu_map(*bo, cpu);
    if (r)
        goto err_map;

    return 0;

err_map:
    amdgpu_bo_va_op(*bo, 0, size, *va, 0, AMDGPU_VA_OP_UNMAP);
err_va:
    amdgpu_va_range_free(*va_handle);
    *va_handle = NULL;
err_bo:
    amdgpu_bo_free(*bo);
    *bo = NULL;
    return r;
}

static void free_bo_and_unmap(amdgpu_bo_handle bo, void *cpu,
                              uint64_t size, uint64_t va,
                              amdgpu_va_handle va_handle)
{
    if (!bo)
        return;

    if (cpu)
        amdgpu_bo_cpu_unmap(bo);
    amdgpu_bo_va_op(bo, 0, size, va, 0, AMDGPU_VA_OP_UNMAP);
    if (va_handle)
        amdgpu_va_range_free(va_handle);
    amdgpu_bo_free(bo);
}

static int submit_copy(amdgpu_device_handle dev, amdgpu_context_handle ctx,
                       amdgpu_bo_handle src, uint64_t src_va,
                       amdgpu_bo_handle dst, uint64_t dst_va,
                       amdgpu_bo_handle cmd_bo, uint64_t cmd_va,
                       void *cmd_cpu, uint64_t size)
{
    uint32_t ib[64];
    unsigned n = 0;
    struct amdgpu_cs_ib_info ib_info = {0};
    struct amdgpu_cs_request req = {0};
    amdgpu_bo_handle resources[3];
    amdgpu_bo_list_handle list = NULL;
    struct amdgpu_cs_fence fence = {0};
    uint32_t expired = 0;
    int r;

    while (size) {
        uint64_t bytes = size > SDMA_MAX_COPY ? SDMA_MAX_COPY : size;

        if (bytes == 0 ||
            n + SDMA_COPY_DWORDS > sizeof(ib) / sizeof(ib[0]))
            return -E2BIG;

        ib[n++] = SDMA_PACKET(SDMA_OPCODE_COPY, SDMA_COPY_SUB_OPCODE_LINEAR, 0);
        ib[n++] = (uint32_t)(bytes - 1);
        ib[n++] = 0;
        ib[n++] = (uint32_t)src_va;
        ib[n++] = (uint32_t)(src_va >> 32);
        ib[n++] = (uint32_t)dst_va;
        ib[n++] = (uint32_t)(dst_va >> 32);
        src_va += bytes;
        dst_va += bytes;
        size -= bytes;
    }

    memcpy(cmd_cpu, ib, n * sizeof(uint32_t));
    ib_info.ib_mc_address = cmd_va;
    ib_info.size = n;

    resources[0] = src;
    resources[1] = dst;
    resources[2] = cmd_bo;
    r = amdgpu_bo_list_create(dev, 3, resources, NULL, &list);
    if (r)
        return r;

    req.ip_type = AMDGPU_HW_IP_DMA;
    req.ip_instance = 0;
    req.ring = 0;
    req.number_of_ibs = 1;
    req.ibs = &ib_info;
    req.resources = list;

    r = amdgpu_cs_submit(ctx, 0, &req, 1);
    if (r)
        goto out_list;

    fence.ip_type = AMDGPU_HW_IP_DMA;
    fence.ip_instance = 0;
    fence.ring = 0;
    fence.context = ctx;
    fence.fence = req.seq_no;
    r = amdgpu_cs_query_fence_status(&fence, AMDGPU_TIMEOUT_NS, 0, &expired);
    if (!r && !expired)
        r = -ETIMEDOUT;

out_list:
    amdgpu_bo_list_destroy(list);
    return r;
}

int main(int argc, char **argv)
{
    const char *op;
    size_t size;
    int dmabuf_fd;
    unsigned pattern;
    int drm_fd = -1, r = 1;
    uint32_t major = 0, minor = 0;
    amdgpu_device_handle dev = NULL;
    amdgpu_context_handle ctx = NULL;
    amdgpu_bo_handle imported = NULL, staging = NULL, cmd_bo = NULL;
    struct amdgpu_bo_import_result imported_info = {0};
    void *staging_cpu = NULL, *cmd_cpu = NULL;
    uint64_t staging_va = 0, imported_va = 0, cmd_va = 0;
    amdgpu_va_handle staging_va_handle = NULL;
    amdgpu_va_handle imported_va_handle = NULL;
    amdgpu_va_handle cmd_va_handle = NULL;

    if (argc != 5)
        return 77;

    op = argv[1];
    size = strtoull(argv[2], NULL, 0);
    dmabuf_fd = atoi(argv[3]);
    pattern = strtoul(argv[4], NULL, 0) & 0xff;
    if (!size || size > (1ULL << 30) ||
        (strcmp(op, "write") && strcmp(op, "read")))
        return 77;

    size = (size + PAGE_ALIGN - 1) & ~(PAGE_ALIGN - 1);

    drm_fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    if (drm_fd < 0)
        return 77;

    r = amdgpu_device_initialize(drm_fd, &major, &minor, &dev);
    if (r)
        goto out;

    r = amdgpu_cs_ctx_create(dev, &ctx);
    if (r)
        goto out;

    r = amdgpu_bo_import(dev, amdgpu_bo_handle_type_dma_buf_fd,
                         (uint32_t)dmabuf_fd, &imported_info);
    if (r)
        goto out;
    imported = imported_info.buf_handle;
    if (!imported)
        goto out;

    r = amdgpu_va_range_alloc(dev, amdgpu_gpu_va_range_general,
                              size, PAGE_ALIGN, 0, &imported_va,
                              &imported_va_handle, 0);
    if (r)
        goto out;

    r = amdgpu_bo_va_op(imported, 0, size, imported_va, 0,
                        AMDGPU_VA_OP_MAP);
    if (r)
        goto out;

    r = alloc_bo_and_map(dev, size, &staging, &staging_cpu,
                         &staging_va, &staging_va_handle);
    if (r)
        goto out;

    r = alloc_bo_and_map(dev, PAGE_ALIGN, &cmd_bo, &cmd_cpu,
                         &cmd_va, &cmd_va_handle);
    if (r)
        goto out;

    if (!strcmp(op, "write")) {
        memset(staging_cpu, pattern, size);
        r = submit_copy(dev, ctx, staging, staging_va, imported, imported_va,
                        cmd_bo, cmd_va, cmd_cpu, size);
        if (!r)
            printf("GPU DMA-BUF WRITE PASS\n");
    } else {
        memset(staging_cpu, 0, size);
        r = submit_copy(dev, ctx, imported, imported_va, staging, staging_va,
                        cmd_bo, cmd_va, cmd_cpu, size);
        if (!r) {
            uint8_t *p = staging_cpu;
            for (size_t i = 0; i < size; i += PAGE_ALIGN) {
                if (p[i] != (uint8_t)pattern) {
                    r = -EIO;
                    break;
                }
            }
        }
        if (!r)
            printf("GPU DMA-BUF READ PASS\n");
    }

out:
    if (imported && imported_va_handle) {
        amdgpu_bo_va_op(imported, 0, size, imported_va, 0,
                        AMDGPU_VA_OP_UNMAP);
        amdgpu_va_range_free(imported_va_handle);
    }
    free_bo_and_unmap(cmd_bo, cmd_cpu, PAGE_ALIGN, cmd_va,
                      cmd_va_handle);
    free_bo_and_unmap(staging, staging_cpu, size, staging_va,
                      staging_va_handle);
    if (imported)
        amdgpu_bo_free(imported);
    if (ctx)
        amdgpu_cs_ctx_free(ctx);
    if (dev)
        amdgpu_device_deinitialize(dev);
    if (drm_fd >= 0)
        close(drm_fd);
    return r ? 1 : 0;
}

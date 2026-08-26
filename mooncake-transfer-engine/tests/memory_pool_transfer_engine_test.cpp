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

constexpr size_t kTransferSize = 2 * 1024 * 1024;

std::string EnvOrDefault(const char *name, const char *fallback)
{
    const char *value = getenv(name);
    if (value != nullptr && *value != '\0') {
        return std::string(value);
    }

    return std::string(fallback);
}

std::string GetDevicePaths()
{
    return EnvOrDefault(
        "MOONCAKE_MEMORY_POOL_DEVICES",
        EnvOrDefault("MOONCAKE_MEMORY_POOL_DEVICE", "/dev/amdgpu-mpu0").c_str());
}

std::vector<std::string> SplitDevices(const std::string &devices)
{
    std::vector<std::string> result;
    std::stringstream stream(devices);
    std::string device;

    while (std::getline(stream, device, ',')) {
        const size_t begin = device.find_first_not_of(" \t\r\n");
        if (begin == std::string::npos) {
            continue;
        }

        const size_t end = device.find_last_not_of(" \t\r\n");
        result.push_back(device.substr(begin, end - begin + 1));
    }

    return result;
}

MemoryPoolTransferEngine OpenEngine()
{
    const std::string sueverbs_library = EnvOrDefault(
        "MOONCAKE_SUEVERBS_LIBRARY", "libsueverbs.so");
    const std::string device_paths = GetDevicePaths();

    return MemoryPoolTransferEngine(sueverbs_library, device_paths);
}

void RequireEngineOpen(MemoryPoolTransferEngine &engine)
{
    if (engine.NodeCount() == 0) {
        GTEST_SKIP() << "No MPU devices configured";
    }

    const std::vector<std::string> devices = SplitDevices(GetDevicePaths());
    for (const std::string &device : devices) {
        if (access(device.c_str(), R_OK | W_OK) != 0) {
            GTEST_SKIP() << "MPU device unavailable: " << device;
        }
    }

    const int ret = engine.Open();
    if (ret != 0) {
        GTEST_SKIP() << "Unable to open MPU backend, rc=" << ret;
    }

    ASSERT_TRUE(engine.IsOpen());
}

}  // namespace

TEST(MemoryPoolTransferEngineTest, MultiNodeAllocationAndTargetRange)
{
    auto engine = OpenEngine();
    RequireEngineOpen(engine);

    ASSERT_GT(engine.Capacity(), 0u);

    std::vector<MemoryPoolTransferEngine::Allocation> allocations;
    allocations.reserve(engine.NodeCount() + 1);

    for (size_t i = 0; i < engine.NodeCount() + 1; ++i) {
        MemoryPoolTransferEngine::Allocation allocation;

        ASSERT_EQ(engine.Allocate(kTransferSize, &allocation), 0);
        ASSERT_TRUE(allocation.valid());

        uint64_t target_address = 0;
        ASSERT_EQ(
            engine.TargetRange(
                allocation, 0, kTransferSize, &target_address),
            0);
        ASSERT_NE(target_address, 0u);

        allocations.push_back(std::move(allocation));
    }

    for (auto &allocation : allocations) {
        ASSERT_EQ(engine.Free(&allocation), 0);
    }
}

TEST(MemoryPoolTransferEngineTest, DmaBufExportAndImport)
{
    auto engine = OpenEngine();
    RequireEngineOpen(engine);

    MemoryPoolTransferEngine::Allocation allocation;
    ASSERT_EQ(engine.Allocate(kTransferSize, &allocation), 0);

    int dma_buf_fd = -1;
    ASSERT_EQ(
        engine.ExportDmaBuf(&allocation, O_CLOEXEC, &dma_buf_fd),
        0);

    MemoryPoolTransferEngine::ImportedDmaBuf gpu_dma_buf;
    MemoryPoolTransferEngine::ImportedDmaBuf nic_dma_buf;

    ASSERT_EQ(
        engine.ImportDmaBuf(
            dma_buf_fd,
            0x10000000,
            kTransferSize,
            MemoryPoolTransferEngine::DmaBufType::GPU,
            &gpu_dma_buf),
        0);

    ASSERT_EQ(
        engine.ImportDmaBuf(
            dma_buf_fd,
            0x20000000,
            kTransferSize,
            MemoryPoolTransferEngine::DmaBufType::NIC,
            &nic_dma_buf),
        0);

    ASSERT_NE(gpu_dma_buf.fd, nic_dma_buf.fd);

    ASSERT_EQ(engine.ReleaseDmaBuf(&gpu_dma_buf), 0);
    ASSERT_EQ(engine.ReleaseDmaBuf(&nic_dma_buf), 0);

    close(dma_buf_fd);

    ASSERT_EQ(engine.Free(&allocation), 0);
}

TEST(MemoryPoolTransferEngineTest, SueQueueSubmission)
{
    auto engine = OpenEngine();
    RequireEngineOpen(engine);

    MemoryPoolTransferEngine::Allocation source;
    MemoryPoolTransferEngine::Allocation destination;

    ASSERT_EQ(engine.Allocate(kTransferSize, &source), 0);
    ASSERT_EQ(engine.Allocate(kTransferSize, &destination), 0);

    int dma_buf_fd = -1;
    ASSERT_EQ(
        engine.ExportDmaBuf(&source, O_CLOEXEC, &dma_buf_fd),
        0);

    uint64_t target_address = 0;
    ASSERT_EQ(
        engine.TargetRange(
            destination, 0, kTransferSize, &target_address),
        0);

    uint64_t cookie = 0;
    const int submit_ret = engine.SubmitDmaBufTransfer(
        dma_buf_fd,
        target_address,
        0,
        kTransferSize,
        AMDGPU_MPU_SUE_OP_READ,
        &cookie);

    ASSERT_EQ(submit_ret, 0);
    ASSERT_NE(cookie, 0u);

    amdgpu_mpu_sue_status_t status{};
    const int status_ret =
        engine.GetDmaBufTransferStatus(cookie, &status);

    ASSERT_TRUE(status_ret == 0 || status_ret == -11);

    close(dma_buf_fd);

    ASSERT_EQ(engine.Free(&source), 0);
    ASSERT_EQ(engine.Free(&destination), 0);
}

TEST(MemoryPoolTransferEngineTest, ExternalDmaBufImport)
{
    const char *gpu_dma_buf_fd = getenv("MOONCAKE_GPU_DMABUF_FD");
    const char *nic_dma_buf_fd = getenv("MOONCAKE_NIC_DMABUF_FD");

    if (gpu_dma_buf_fd == nullptr && nic_dma_buf_fd == nullptr) {
        GTEST_SKIP()
            << "Set MOONCAKE_GPU_DMABUF_FD and/or "
               "MOONCAKE_NIC_DMABUF_FD";
    }

    auto engine = OpenEngine();
    RequireEngineOpen(engine);

    std::vector<MemoryPoolTransferEngine::ImportedDmaBuf> imported_buffers;

    auto import_dma_buf = [&](const char *fd_string,
                              uint64_t address,
                              MemoryPoolTransferEngine::DmaBufType type) {
        if (fd_string == nullptr) {
            return;
        }

        char *end = nullptr;
        const long fd = strtol(fd_string, &end, 10);

        ASSERT_TRUE(fd_string != end && *end == '\0');

        MemoryPoolTransferEngine::ImportedDmaBuf imported;
        ASSERT_EQ(
            engine.ImportDmaBuf(
                static_cast<int>(fd),
                address,
                kTransferSize,
                type,
                &imported),
            0);

        imported_buffers.push_back(std::move(imported));
    };

    import_dma_buf(
        gpu_dma_buf_fd,
        0x30000000,
        MemoryPoolTransferEngine::DmaBufType::GPU);

    import_dma_buf(
        nic_dma_buf_fd,
        0x40000000,
        MemoryPoolTransferEngine::DmaBufType::NIC);

    for (auto &imported : imported_buffers) {
        ASSERT_EQ(engine.ReleaseDmaBuf(&imported), 0);
    }
}

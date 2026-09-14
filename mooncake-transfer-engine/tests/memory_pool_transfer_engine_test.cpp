// Copyright 2026 Mooncake Authors
// Licensed under the Apache License, Version 2.0

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <limits>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "memory_pool_transfer_engine.h"

using namespace mooncake;

namespace {

constexpr uint64_t kPageSize = 4096;
constexpr uint64_t kAllocationSize = 2 * 1024 * 1024;

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
    const char *devices = getenv("MOONCAKE_MEMORY_POOL_DEVICES");
    if (devices != nullptr && *devices != '\0') {
        return std::string(devices);
    }

    return EnvOrDefault("MOONCAKE_MEMORY_POOL_DEVICE", "/dev/amdgpu-mpu0");
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
    // Retained for constructor compatibility. Passive-MPU tests do not use
    // the MPU-initiated SUE submission path.
    const std::string sueverbs_library = EnvOrDefault(
        "MOONCAKE_SUEVERBS_LIBRARY", "libsueverbs.so");

    return MemoryPoolTransferEngine(sueverbs_library, GetDevicePaths());
}

void RequireEngineOpen(MemoryPoolTransferEngine &engine)
{
    if (engine.NodeCount() == 0) {
        GTEST_SKIP() << "No MPU devices configured";
    }

    for (const std::string &device : SplitDevices(GetDevicePaths())) {
        if (access(device.c_str(), R_OK | W_OK) != 0) {
            GTEST_SKIP() << "MPU device unavailable: " << device;
        }
    }

    const int ret = engine.Open();
    if (ret != 0) {
        GTEST_SKIP() << "Unable to open MPU backend, rc=" << ret;
    }

    ASSERT_TRUE(engine.IsOpen());
    ASSERT_GT(engine.NodeCount(), 0u);
    ASSERT_GT(engine.Capacity(), 0u);
}

void RequireValidAllocation(
    const MemoryPoolTransferEngine::Allocation &allocation)
{
    ASSERT_TRUE(allocation.valid());
    ASSERT_FALSE(allocation.mapped());
}

void TestExternalDmaBufImport(
    MemoryPoolTransferEngine &engine,
    const char *environment_name,
    uint64_t expected_address,
    MemoryPoolTransferEngine::DmaBufType type)
{
    const char *fd_string = getenv(environment_name);
    if (fd_string == nullptr || *fd_string == '\0') {
        GTEST_SKIP() << "Set " << environment_name
                     << " to test external DMA-BUF import";
    }

    char *end = nullptr;
    errno = 0;
    const long parsed_fd = strtol(fd_string, &end, 10);

    ASSERT_EQ(errno, 0);
    ASSERT_NE(end, fd_string);
    ASSERT_EQ(*end, '\0');
    ASSERT_GE(parsed_fd, 0);
    ASSERT_LE(parsed_fd, std::numeric_limits<int>::max());

    MemoryPoolTransferEngine::ImportedDmaBuf imported;

    ASSERT_EQ(
        engine.ImportDmaBuf(
            static_cast<int>(parsed_fd),
            expected_address,
            kAllocationSize,
            type,
            &imported),
        0);

    ASSERT_TRUE(imported.valid());
    ASSERT_EQ(imported.length, kAllocationSize);
    ASSERT_EQ(imported.address, expected_address);
    ASSERT_EQ(imported.type, type);

    ASSERT_EQ(engine.ReleaseDmaBuf(&imported), 0);
    ASSERT_FALSE(imported.valid());
}

}  // namespace

TEST(MemoryPoolTransferEngineTest, OpenAndDiscoverNodes)
{
    auto engine = OpenEngine();
    RequireEngineOpen(engine);

    for (size_t node_id = 0; node_id < engine.NodeCount(); ++node_id) {
        ASSERT_GT(engine.NodeCapacity(static_cast<uint32_t>(node_id)), 0u);
    }
}

TEST(MemoryPoolTransferEngineTest, AllocateTargetRangeAndFree)
{
    auto engine = OpenEngine();
    RequireEngineOpen(engine);

    MemoryPoolTransferEngine::Allocation allocation;

    ASSERT_EQ(engine.Allocate(kAllocationSize, &allocation), 0);
    RequireValidAllocation(allocation);

    uint64_t target_address = 0;
    ASSERT_EQ(
        engine.TargetRange(
            allocation,
            0,
            kAllocationSize,
            &target_address),
        0);
    ASSERT_NE(target_address, 0u);

    ASSERT_EQ(engine.Free(&allocation), 0);
    ASSERT_FALSE(allocation.valid());
}

TEST(MemoryPoolTransferEngineTest, AllocateSubrangesWithinOneAllocation)
{
    auto engine = OpenEngine();
    RequireEngineOpen(engine);

    MemoryPoolTransferEngine::Allocation allocation;
    ASSERT_EQ(engine.Allocate(kAllocationSize, &allocation), 0);
    RequireValidAllocation(allocation);

    uint64_t first_address = 0;
    uint64_t second_address = 0;

    ASSERT_EQ(
        engine.TargetRange(allocation, 0, kPageSize, &first_address),
        0);
    ASSERT_EQ(
        engine.TargetRange(
            allocation, kPageSize, kPageSize, &second_address),
        0);

    ASSERT_NE(first_address, 0u);
    ASSERT_NE(second_address, 0u);
    ASSERT_EQ(second_address - first_address, kPageSize);

    ASSERT_EQ(engine.Free(&allocation), 0);
}

TEST(MemoryPoolTransferEngineTest, RejectInvalidTargetRanges)
{
    auto engine = OpenEngine();
    RequireEngineOpen(engine);

    MemoryPoolTransferEngine::Allocation allocation;
    ASSERT_EQ(engine.Allocate(kAllocationSize, &allocation), 0);
    RequireValidAllocation(allocation);

    uint64_t target_address = 0;

    ASSERT_NE(
        engine.TargetRange(allocation, 0, 0, &target_address),
        0);
    ASSERT_NE(
        engine.TargetRange(
            allocation, kAllocationSize + 1, 1, &target_address),
        0);
    ASSERT_NE(
        engine.TargetRange(
            allocation, kAllocationSize - 1, 2, &target_address),
        0);

    ASSERT_EQ(engine.Free(&allocation), 0);
}

TEST(MemoryPoolTransferEngineTest, ExportMemoryPoolDmaBuf)
{
    auto engine = OpenEngine();
    RequireEngineOpen(engine);

    MemoryPoolTransferEngine::Allocation allocation;
    ASSERT_EQ(engine.Allocate(kAllocationSize, &allocation), 0);
    RequireValidAllocation(allocation);

    int dma_buf_fd = -1;
    ASSERT_EQ(
        engine.ExportDmaBuf(&allocation, O_CLOEXEC, &dma_buf_fd),
        0);
    ASSERT_GE(dma_buf_fd, 0);

    ASSERT_EQ(close(dma_buf_fd), 0);
    ASSERT_EQ(engine.Free(&allocation), 0);
}

TEST(MemoryPoolTransferEngineTest, ImportExportedDmaBufForGpuAndNic)
{
    auto engine = OpenEngine();
    RequireEngineOpen(engine);

    MemoryPoolTransferEngine::Allocation allocation;
    ASSERT_EQ(engine.Allocate(kAllocationSize, &allocation), 0);
    RequireValidAllocation(allocation);

    int exported_fd = -1;
    ASSERT_EQ(
        engine.ExportDmaBuf(&allocation, O_CLOEXEC, &exported_fd),
        0);
    ASSERT_GE(exported_fd, 0);

    MemoryPoolTransferEngine::ImportedDmaBuf gpu_buffer;
    MemoryPoolTransferEngine::ImportedDmaBuf nic_buffer;

    ASSERT_EQ(
        engine.ImportDmaBuf(
            exported_fd,
            0x10000000,
            kAllocationSize,
            MemoryPoolTransferEngine::DmaBufType::GPU,
            &gpu_buffer),
        0);
    ASSERT_EQ(
        engine.ImportDmaBuf(
            exported_fd,
            0x20000000,
            kAllocationSize,
            MemoryPoolTransferEngine::DmaBufType::NIC,
            &nic_buffer),
        0);

    ASSERT_TRUE(gpu_buffer.valid());
    ASSERT_TRUE(nic_buffer.valid());
    ASSERT_NE(gpu_buffer.fd, nic_buffer.fd);

    ASSERT_EQ(engine.ReleaseDmaBuf(&gpu_buffer), 0);
    ASSERT_EQ(engine.ReleaseDmaBuf(&nic_buffer), 0);
    ASSERT_FALSE(gpu_buffer.valid());
    ASSERT_FALSE(nic_buffer.valid());

    ASSERT_EQ(close(exported_fd), 0);
    ASSERT_EQ(engine.Free(&allocation), 0);
}

TEST(MemoryPoolTransferEngineTest, ImportExternalGpuDmaBuf)
{
    auto engine = OpenEngine();
    RequireEngineOpen(engine);

    TestExternalDmaBufImport(
        engine,
        "MOONCAKE_GPU_DMABUF_FD",
        0x30000000,
        MemoryPoolTransferEngine::DmaBufType::GPU);
}

TEST(MemoryPoolTransferEngineTest, ImportExternalNicDmaBuf)
{
    auto engine = OpenEngine();
    RequireEngineOpen(engine);

    TestExternalDmaBufImport(
        engine,
        "MOONCAKE_NIC_DMABUF_FD",
        0x40000000,
        MemoryPoolTransferEngine::DmaBufType::NIC);
}

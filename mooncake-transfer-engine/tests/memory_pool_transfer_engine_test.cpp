// Copyright 2026 Mooncake Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

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

std::string EnvOrDefault(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value && *value ? std::string(value) : std::string(fallback);
}

std::vector<std::string> SplitDevices(const std::string& devices) {
    std::vector<std::string> result;
    std::stringstream stream(devices);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const auto first = item.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        const auto last = item.find_last_not_of(" \t\r\n");
        result.push_back(item.substr(first, last - first + 1));
    }
    return result;
}

MemoryPoolTransferEngine OpenEngine() {
    const std::string device_paths =
        EnvOrDefault("MOONCAKE_MEMORY_POOL_DEVICES",
                     EnvOrDefault("MOONCAKE_MEMORY_POOL_DEVICE",
                                  "/dev/amdgpu-mpu0"));
    const std::string sueverbs_library =
        EnvOrDefault("MOONCAKE_SUEVERBS_LIBRARY", "libsueverbs.so");
    return MemoryPoolTransferEngine(sueverbs_library, device_paths);
}

void RequireEngineOpen(MemoryPoolTransferEngine& engine) {
    if (engine.NodeCount() == 0) GTEST_SKIP() << "No MPU devices configured";

    const std::string device_paths =
        EnvOrDefault("MOONCAKE_MEMORY_POOL_DEVICES",
                     EnvOrDefault("MOONCAKE_MEMORY_POOL_DEVICE",
                                  "/dev/amdgpu-mpu0"));
    for (const auto& device : SplitDevices(device_paths)) {
        if (access(device.c_str(), R_OK | W_OK) != 0)
            GTEST_SKIP() << "MPU device is unavailable: " << device;
    }

    const int rc = engine.Open();
    if (rc != 0) GTEST_SKIP() << "Unable to open MPU backend, rc=" << rc;
    ASSERT_TRUE(engine.IsOpen());
}
}  // namespace

TEST(MemoryPoolTransferEngineTest, MultiNodeAllocationAndTargetRange) {
    auto engine = OpenEngine();
    RequireEngineOpen(engine);

    ASSERT_GT(engine.Capacity(), 0u);

    std::vector<MemoryPoolTransferEngine::Allocation> allocations;
    allocations.reserve(engine.NodeCount() + 1);
    for (size_t i = 0; i < engine.NodeCount() + 1; ++i) {
        MemoryPoolTransferEngine::Allocation allocation{};
        ASSERT_EQ(engine.Allocate(kTransferSize, &allocation), 0);
        ASSERT_TRUE(allocation.valid());
        ASSERT_GE(allocation.buf.size, kTransferSize);
        ASSERT_EQ(allocation.node_id, i % engine.NodeCount());
        ASSERT_GT(engine.NodeCapacity(allocation.node_id), 0u);

        uint64_t target_addr = 0;
        ASSERT_EQ(engine.TargetRange(allocation, 0, kTransferSize,
                                     &target_addr),
                  0);
        ASSERT_NE(target_addr, 0u);
        allocations.push_back(std::move(allocation));
    }

    for (auto& allocation : allocations) {
        ASSERT_EQ(engine.Free(&allocation), 0);
    }
}

TEST(MemoryPoolTransferEngineTest, DmaBufExportAndImport) {
    auto engine = OpenEngine();
    RequireEngineOpen(engine);

    MemoryPoolTransferEngine::Allocation pool{};
    ASSERT_EQ(engine.Allocate(kTransferSize, &pool), 0);

    int exported_fd = -1;
    ASSERT_EQ(engine.ExportDmaBuf(&pool, O_CLOEXEC, &exported_fd), 0);
    ASSERT_GE(exported_fd, 0);

    // Exercise the common importer with a DMA-BUF produced by the MPU itself.
    // This validates fd ownership/lifetime without pretending that a kernel
    // DMA-BUF attachment has happened. Real GPU/NIC exporters can be supplied
    // through the environment in the external-import test below.
    MemoryPoolTransferEngine::ImportedDmaBuf gpu_import{};
    ASSERT_EQ(engine.ImportDmaBuf(exported_fd, 0x10000000, kTransferSize,
                                  MemoryPoolTransferEngine::DmaBufType::GPU,
                                  &gpu_import),
              0);
    ASSERT_TRUE(gpu_import.valid());
    ASSERT_EQ(gpu_import.length, kTransferSize);
    ASSERT_EQ(gpu_import.address, 0x10000000u);
    ASSERT_EQ(gpu_import.type, MemoryPoolTransferEngine::DmaBufType::GPU);
    ASSERT_NE(gpu_import.fd, exported_fd);

    MemoryPoolTransferEngine::ImportedDmaBuf nic_import{};
    ASSERT_EQ(engine.ImportDmaBuf(exported_fd, 0x20000000, kTransferSize,
                                  MemoryPoolTransferEngine::DmaBufType::NIC,
                                  &nic_import),
              0);
    ASSERT_TRUE(nic_import.valid());
    ASSERT_EQ(nic_import.length, kTransferSize);
    ASSERT_EQ(nic_import.address, 0x20000000u);
    ASSERT_EQ(nic_import.type, MemoryPoolTransferEngine::DmaBufType::NIC);
    ASSERT_NE(nic_import.fd, exported_fd);
    ASSERT_NE(nic_import.fd, gpu_import.fd);

    ASSERT_EQ(engine.ReleaseDmaBuf(&gpu_import), 0);
    ASSERT_FALSE(gpu_import.valid());
    ASSERT_EQ(engine.ReleaseDmaBuf(&nic_import), 0);
    ASSERT_FALSE(nic_import.valid());

    close(exported_fd);
    ASSERT_EQ(engine.Free(&pool), 0);
}

TEST(MemoryPoolTransferEngineTest, ExternalDmaBufImport) {
    const char* gpu_fd_env = std::getenv("MOONCAKE_GPU_DMABUF_FD");
    const char* nic_fd_env = std::getenv("MOONCAKE_NIC_DMABUF_FD");
    if (!gpu_fd_env && !nic_fd_env) {
        GTEST_SKIP() << "Set MOONCAKE_GPU_DMABUF_FD and/or "
                        "MOONCAKE_NIC_DMABUF_FD to test external exporters";
    }

    auto engine = OpenEngine();
    RequireEngineOpen(engine);

    std::vector<MemoryPoolTransferEngine::ImportedDmaBuf> imports;
    auto import_external = [&](const char* env, uint64_t address,
                               MemoryPoolTransferEngine::DmaBufType type) {
        if (!env) return;
        char* end = nullptr;
        const long fd = std::strtol(env, &end, 10);
        ASSERT_TRUE(end != env && *end == '\0')
            << "Invalid DMA-BUF fd: " << env;
        ASSERT_GE(fd, 0);

        MemoryPoolTransferEngine::ImportedDmaBuf imported{};
        ASSERT_EQ(engine.ImportDmaBuf(static_cast<int>(fd), address,
                                      kTransferSize, type, &imported),
                  0);
        ASSERT_TRUE(imported.valid());
        ASSERT_EQ(imported.length, kTransferSize);
        ASSERT_EQ(imported.address, address);
        ASSERT_EQ(imported.type, type);
        imports.push_back(std::move(imported));
    };

    import_external(gpu_fd_env, 0x30000000,
                    MemoryPoolTransferEngine::DmaBufType::GPU);
    import_external(nic_fd_env, 0x40000000,
                    MemoryPoolTransferEngine::DmaBufType::NIC);

    for (auto& imported : imports) {
        ASSERT_EQ(engine.ReleaseDmaBuf(&imported), 0);
        ASSERT_FALSE(imported.valid());
    }
}

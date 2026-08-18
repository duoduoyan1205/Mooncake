// Copyright 2026 Mooncake Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <gtest/gtest.h>

#include <cstdlib>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "cuda_alike.h"
#include "memory_pool_transfer_engine.h"

using namespace mooncake;

namespace {
constexpr size_t kTransferSize = 2 * 1024 * 1024;
constexpr unsigned char kPPattern = 0x5a;
constexpr unsigned char kDPattern = 0xa5;

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

class DeviceBuffer {
 public:
    DeviceBuffer(int device, size_t size) : device_(device), size_(size) {
        ASSERT_EQ(cudaSetDevice(device_), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&ptr_, size_), cudaSuccess);
    }

    ~DeviceBuffer() {
        if (ptr_) {
            cudaSetDevice(device_);
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* get() const { return ptr_; }

    void Fill(unsigned char value) {
        ASSERT_EQ(cudaSetDevice(device_), cudaSuccess);
        ASSERT_EQ(cudaMemset(ptr_, value, size_), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    std::vector<unsigned char> ReadBack() const {
        std::vector<unsigned char> host(size_);
        EXPECT_EQ(cudaSetDevice(device_), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(host.data(), ptr_, size_, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return host;
    }

 private:
    int device_;
    size_t size_;
    void* ptr_ = nullptr;
};

void ExpectPattern(const std::vector<unsigned char>& data, unsigned char value) {
    ASSERT_EQ(data.size(), kTransferSize);
    for (size_t i = 0; i < data.size(); ++i) {
        ASSERT_EQ(data[i], value) << "data mismatch at byte " << i;
    }
}
}  // namespace

TEST(MemoryPoolTransferEngineTest, FourAccessPathsMultiNode) {
    const std::string device_paths =
        EnvOrDefault("MOONCAKE_MEMORY_POOL_DEVICES",
                     EnvOrDefault("MOONCAKE_MEMORY_POOL_DEVICE",
                                  "/dev/amdgpu-mpu0"));
    const std::string sueverbs_library =
        EnvOrDefault("MOONCAKE_SUEVERBS_LIBRARY", "libsueverbs.so");
    const auto devices = SplitDevices(device_paths);
    if (devices.empty()) GTEST_SKIP() << "No MPU devices configured";

    for (const auto& device : devices) {
        if (access(device.c_str(), R_OK | W_OK) != 0) {
            GTEST_SKIP() << "MPU device is unavailable: " << device;
        }
    }

    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count < 2) {
        GTEST_SKIP() << "P->D/D->P test requires at least two GPUs";
    }

    MemoryPoolTransferEngine engine(sueverbs_library, device_paths);
    const int open_rc = engine.Open();
    if (open_rc != 0) {
        GTEST_SKIP() << "Unable to open MPU/SUE verbs backend, rc=" << open_rc
                     << ", library=" << sueverbs_library;
    }

    ASSERT_TRUE(engine.IsOpen());
    ASSERT_EQ(engine.NodeCount(), devices.size());
    ASSERT_GT(engine.Capacity(), 0u);

    DeviceBuffer p_buffer(0, kTransferSize);
    DeviceBuffer d_buffer(1, kTransferSize);

    // P -> D and D -> P use the same MPU topology. Node 0 is sufficient for
    // this directional pair; the Memory Pool placement itself is multi-node.
    p_buffer.Fill(kPPattern);
    d_buffer.Fill(0);
    ASSERT_EQ(engine.TransferPToD(0, reinterpret_cast<uint64_t>(p_buffer.get()),
                                  reinterpret_cast<uint64_t>(d_buffer.get()),
                                  kTransferSize),
              0);
    ExpectPattern(d_buffer.ReadBack(), kPPattern);

    d_buffer.Fill(kDPattern);
    p_buffer.Fill(0);
    ASSERT_EQ(engine.TransferDToP(0, reinterpret_cast<uint64_t>(d_buffer.get()),
                                  reinterpret_cast<uint64_t>(p_buffer.get()),
                                  kTransferSize),
              0);
    ExpectPattern(p_buffer.ReadBack(), kDPattern);

    // Allocate one object per configured Memory Pool node plus one extra.
    // The manager must assign node ids strictly round-robin.
    std::vector<MemoryPoolTransferEngine::Allocation> allocations;
    allocations.reserve(devices.size() + 1);
    for (size_t i = 0; i < devices.size() + 1; ++i) {
        MemoryPoolTransferEngine::Allocation allocation{};
        ASSERT_EQ(engine.Allocate(kTransferSize, &allocation), 0);
        ASSERT_NE(allocation.handle, 0u);
        ASSERT_NE(allocation.global_addr, 0u);
        ASSERT_GE(allocation.size, kTransferSize);
        ASSERT_EQ(allocation.node_id, i % devices.size());
        ASSERT_GT(engine.NodeCapacity(allocation.node_id), 0u);
        allocations.push_back(allocation);
    }

    // Validate D -> Pool and Pool -> D independently on every node.
    for (const auto& allocation : allocations) {
        d_buffer.Fill(kDPattern);
        ASSERT_EQ(engine.TransferDToPool(allocation,
                                         reinterpret_cast<uint64_t>(d_buffer.get()),
                                         kTransferSize),
                  0);

        d_buffer.Fill(0);
        ASSERT_EQ(engine.TransferPoolToD(
                      allocation, reinterpret_cast<uint64_t>(d_buffer.get()),
                      kTransferSize),
                  0);
        ExpectPattern(d_buffer.ReadBack(), kDPattern);
    }

    for (const auto& allocation : allocations) {
        ASSERT_EQ(engine.Free(allocation), 0);
    }
}

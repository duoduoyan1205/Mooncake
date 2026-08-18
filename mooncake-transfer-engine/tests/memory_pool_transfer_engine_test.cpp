// Copyright 2026 Mooncake Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <cstdlib>
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

// Integration test for an MPU/SUE-capable node. It intentionally skips when
// the driver, SUE verbs library, or two GPU devices are unavailable. On a
// real node it exercises all four MPU data-plane paths:
//   P -> D, D -> P, D -> Memory Pool, Memory Pool -> D.
TEST(MemoryPoolTransferEngineTest, FourAccessPaths) {
    const std::string device_path =
        EnvOrDefault("MOONCAKE_MEMORY_POOL_DEVICE", "/dev/amdgpu-mpu");
    const std::string sueverbs_library =
        EnvOrDefault("MOONCAKE_SUEVERBS_LIBRARY", "libsueverbs.so");

    if (access(device_path.c_str(), R_OK | W_OK) != 0) {
        GTEST_SKIP() << "MPU device is unavailable: " << device_path;
    }

    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count < 2) {
        GTEST_SKIP() << "MPU P->D/D->P test requires at least two GPUs";
    }

    MemoryPoolTransferEngine engine(sueverbs_library, device_path);
    const int open_rc = engine.Open();
    if (open_rc != 0) {
        GTEST_SKIP() << "Unable to open MPU/SUE verbs backend, rc=" << open_rc
                     << ", library=" << sueverbs_library;
    }

    ASSERT_TRUE(engine.IsOpen());
    ASSERT_GT(engine.Capacity(), 0u);

    // P and D deliberately use different GPU devices. This prevents the
    // direct P/D tests from accidentally becoming same-device copies.
    DeviceBuffer p_buffer(0, kTransferSize);
    DeviceBuffer d_buffer(1, kTransferSize);

    // -----------------------------------------------------------------------
    // 1. P -> D
    // -----------------------------------------------------------------------
    p_buffer.Fill(kPPattern);
    d_buffer.Fill(0);
    ASSERT_EQ(engine.TransferPToD(reinterpret_cast<uint64_t>(p_buffer.get()),
                                  reinterpret_cast<uint64_t>(d_buffer.get()),
                                  kTransferSize),
              0);
    ExpectPattern(d_buffer.ReadBack(), kPPattern);

    // -----------------------------------------------------------------------
    // 2. D -> P
    // -----------------------------------------------------------------------
    d_buffer.Fill(kDPattern);
    p_buffer.Fill(0);
    ASSERT_EQ(engine.TransferDToP(reinterpret_cast<uint64_t>(d_buffer.get()),
                                  reinterpret_cast<uint64_t>(p_buffer.get()),
                                  kTransferSize),
              0);
    ExpectPattern(p_buffer.ReadBack(), kDPattern);

    // Allocate one Memory Pool object and use its global SUE address for both
    // pool directions. No CPU virtual address is used for the pool object.
    MemoryPoolTransferEngine::Allocation pool{};
    ASSERT_EQ(engine.Allocate(kTransferSize, &pool), 0);
    ASSERT_NE(pool.handle, 0u);
    ASSERT_NE(pool.global_addr, 0u);
    ASSERT_GE(pool.size, kTransferSize);

    // -----------------------------------------------------------------------
    // 3. D -> Memory Pool
    // -----------------------------------------------------------------------
    d_buffer.Fill(kDPattern);
    ASSERT_EQ(engine.TransferDToPool(
                  reinterpret_cast<uint64_t>(d_buffer.get()), pool.global_addr,
                  kTransferSize),
              0);

    // -----------------------------------------------------------------------
    // 4. Memory Pool -> D
    // -----------------------------------------------------------------------
    d_buffer.Fill(0);
    ASSERT_EQ(engine.TransferPoolToD(
                  pool.global_addr, reinterpret_cast<uint64_t>(d_buffer.get()),
                  kTransferSize),
              0);
    ExpectPattern(d_buffer.ReadBack(), kDPattern);

    ASSERT_EQ(engine.Free(pool), 0);
}

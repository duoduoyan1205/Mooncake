#pragma once

#include <cstdint>
#include <string>

#include "ylt/struct_pack.hpp"

namespace mooncake {

// Persistent metadata for one allocation in the MPU Memory Pool Node.
// global_address is the initiator-visible target address returned by the MPU;
// gpu_address is the MPU-local GPUVM address and is not a remote GPU pointer.
struct MemoryPoolDescriptor {
    uint64_t allocation_handle = 0;
    uint64_t global_address = 0;
    uint64_t gpu_address = 0;
    uint64_t object_size = 0;
    std::string node_id;
    std::string transport_endpoint;

    YLT_REFL(MemoryPoolDescriptor, allocation_handle, global_address,
             gpu_address, object_size, node_id, transport_endpoint);

    bool IsValid() const {
        return allocation_handle != 0 && global_address != 0 &&
               object_size != 0 && !node_id.empty();
    }
};

}  // namespace mooncake

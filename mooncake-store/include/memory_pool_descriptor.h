#pragma once

#include <cstdint>
#include <string>

#include "ylt/struct_pack.hpp"

namespace mooncake {

// Persistent metadata describing one allocation in the SUE Memory Pool.
// This is deliberately independent of the process-local backend Entry so the
// same structure can later be embedded in Replica::Descriptor and exchanged
// through the Mooncake master.
struct MemoryPoolDescriptor {
    uint64_t allocation_handle = 0;
    uint64_t global_address = 0;
    uint64_t object_size = 0;
    std::string node_id;
    std::string transport_endpoint;

    YLT_REFL(MemoryPoolDescriptor, allocation_handle, global_address,
             object_size, node_id, transport_endpoint);

    bool IsValid() const {
        return allocation_handle != 0 && global_address != 0 &&
               object_size != 0 && !node_id.empty();
    }
};

}  // namespace mooncake

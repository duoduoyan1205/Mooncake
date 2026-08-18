#include "memory_pool_storage_backend.h"

#include <cstdlib>
#include <string>

#include "environ.h"
#include "storage_backend.h"

namespace mooncake {

// storage_backend.cpp is compiled with CreateStorageBackend renamed to
// CreateStorageBackendLegacy. Keeping the legacy factory intact avoids a
// large invasive edit to the mature backend switch while allowing this new
// backend to be added as a first-class runtime-selected implementation.
tl::expected<std::shared_ptr<StorageBackendInterface>, ErrorCode>
CreateStorageBackendLegacy(const FileStorageConfig& config);

tl::expected<std::shared_ptr<StorageBackendInterface>, ErrorCode>
CreateStorageBackend(const FileStorageConfig& config) {
    const auto descriptor = Environ::GetString(
        "MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR", "");
    if (descriptor == "memory_pool_storage_backend") {
        return std::make_shared<MemoryPoolStorageBackend>(config);
    }
    return CreateStorageBackendLegacy(config);
}

}  // namespace mooncake

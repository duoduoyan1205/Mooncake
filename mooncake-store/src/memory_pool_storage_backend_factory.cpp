#include "memory_pool_storage_backend.h"

#include <cstdlib>
#include <string>

#include "storage_backend.h"

namespace mooncake {

// storage_backend.cpp is compiled with CreateStorageBackend renamed to
// CreateStorageBackendLegacy. Keeping the legacy factory intact avoids a
// large invasive edit to the mature backend switch while allowing this new
// backend to be added as a first-class runtime-selected implementation.
tl::expected<std::shared_ptr<StorageBackendInterface>, ErrorCode>
CreateStorageBackendLegacy(const FileStorageConfig& config);

namespace {
std::string GetBackendDescriptor() {
    const char* value =
        std::getenv("MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR");
    return value != nullptr ? value : "";
}
}  // namespace

tl::expected<std::shared_ptr<StorageBackendInterface>, ErrorCode>
CreateStorageBackend(const FileStorageConfig& config) {
    if (GetBackendDescriptor() == "memory_pool_storage_backend") {
        return std::make_shared<MemoryPoolStorageBackend>(config);
    }
    return CreateStorageBackendLegacy(config);
}

}  // namespace mooncake

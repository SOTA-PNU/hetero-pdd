// SPDX-License-Identifier: Apache-2.0
// Optional host-only extension for the validated Linux CA22 RBLN SDK build.
#include <rebel/runtime/api/rbln_runtime_api.h>
#include <dlfcn.h>
#include <link.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>

namespace {
constexpr unsigned char kValidatedBuildId[] = {
    0xec, 0x47, 0xd8, 0xf9, 0xe2, 0xc6, 0x2c, 0xf4, 0xbf, 0x68,
    0x33, 0x5b, 0x23, 0xad, 0x35, 0xf4, 0x11, 0xd8, 0xf3, 0xcc,
};

struct Library {
    uintptr_t base;
    bool matched;
};

int match_build_id(dl_phdr_info* info, size_t, void* arg) {
    auto* lib = static_cast<Library*>(arg);
    if (info->dlpi_addr != lib->base) return 0;
    for (unsigned i = 0; i < info->dlpi_phnum; ++i) {
        const auto& hdr = info->dlpi_phdr[i];
        if (hdr.p_type != PT_NOTE) continue;
        const auto* ptr = reinterpret_cast<const unsigned char*>(lib->base + hdr.p_vaddr);
        const auto* end = ptr + hdr.p_memsz;
        while (static_cast<size_t>(end - ptr) >= sizeof(ElfW(Nhdr))) {
            ElfW(Nhdr) note{};
            std::memcpy(&note, ptr, sizeof(note));
            ptr += sizeof(note);
            const size_t name_size = (size_t(note.n_namesz) + 3) & ~size_t(3);
            const size_t desc_size = (size_t(note.n_descsz) + 3) & ~size_t(3);
            if (name_size + desc_size > static_cast<size_t>(end - ptr)) break;
            if (note.n_type == NT_GNU_BUILD_ID && note.n_namesz == 4 &&
                !std::memcmp(ptr, "GNU", 4) && note.n_descsz == sizeof(kValidatedBuildId) &&
                !std::memcmp(ptr + name_size, kValidatedBuildId, sizeof(kValidatedBuildId))) {
                lib->matched = true;
                return 1;
            }
            ptr += name_size + desc_size;
        }
    }
    return 1;
}
}  // namespace

extern "C" bool ca22_canonical_v1_supported() {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&rbln::rbln_get_memory_info), &info)) return false;
    Library lib{reinterpret_cast<uintptr_t>(info.dli_fbase), false};
    dl_iterate_phdr(match_build_id, &lib);
    return lib.matched;
}

// Only accept unsharded model-shaped allocations. Python checks the actual device
// buffers, their size, tensor contiguity, shape and dtype alongside this metadata.
extern "C" bool ca22_canonical_v1_target(uint64_t addr, uint64_t base, uint64_t size,
                                          uint64_t allocation, uint32_t device) {
    try {
        rbln::MemoryInfo info{};
        if (addr < base || size > allocation || addr - base > allocation - size ||
            rbln::rbln_get_memory_info(addr, info) != RBLNRetCode_SUCCESS) return false;
        if (info.key_vaddr != base || info.torch_device_id != device ||
            info.user_dtype != rbln::DataType::Float16 ||
            info.physical_dtype != rbln::DataType::CustomFloat16 ||
            info.user_shape != info.physical_shape || info.user_shape.size() != 6) return false;
        uint64_t bytes = 2;
        for (auto dim : info.user_shape) {
            if (dim <= 0 || static_cast<uint64_t>(dim) > allocation / bytes) return false;
            bytes *= static_cast<uint64_t>(dim);
        }
        return bytes == allocation;
    } catch (...) {
        return false;
    }
}

extern "C" const char* ca22_canonical_v1_describe(uint64_t addr) {
    static thread_local std::string description;
    try {
        rbln::MemoryInfo info{};
        if (rbln::rbln_get_memory_info(addr, info) != RBLNRetCode_SUCCESS) return "unavailable";
        std::ostringstream out;
        out << "logical_dtype=" << static_cast<int>(info.user_dtype) << " logical_dims=";
        for (auto dim : info.user_shape) out << dim << ',';
        out << " physical_dtype=" << static_cast<int>(info.physical_dtype) << " physical_dims=";
        for (auto dim : info.physical_shape) out << dim << ',';
        description = out.str();
        return description.c_str();
    } catch (...) {
        return "unavailable";
    }
}

extern "C" void ca22_canonical_v1_prepare(const uint16_t* input, uint16_t* output, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const uint16_t x = input[i];
        const uint16_t exponent = x & 0x7c00;
        // Preserve all zeros, subnormals, NaNs, infinities, and the two maximal
        // finite values whose next magnitude is not an IEEE half finite number.
        const bool normal = exponent != 0 && exponent != 0x7c00 && (x & 0x7fff) != 0x7bff;
        output[i] = x + static_cast<uint16_t>(normal && (x & 1));
    }
}

extern "C" void ca22_canonical_v1_pushes(uint64_t* count, uint64_t* bytes) {
    rbln_prof_get_host_sync_h2d(count, bytes);
}

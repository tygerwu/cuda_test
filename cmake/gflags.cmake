if(TARGET gflags OR gflags_POPULATED)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    gflags
    GIT_REPOSITORY https://github.com/gflags/gflags.git
    GIT_TAG e171aa2d15ed9eb17054558e0b3a6a413bb01067 #release-2.22
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
FetchContent_MakeAvailable(gflags)
set(gflags_INCLUDE_DIR ${gflags_BINARY_DIR}/include)
set(gflags_LIBRARIES gflags)


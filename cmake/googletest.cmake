if(TARGET googletest OR googletest_POPULATED)
    return()
endif()

include(FetchContent)
message(VERBOSE "FetchContent_Declare(googletest)")
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG dea0216d0c6bc5e63cf5f6c8651cd268668032ec
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

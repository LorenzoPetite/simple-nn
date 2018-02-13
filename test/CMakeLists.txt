enable_testing()

find_package(Boost COMPONENTS unit_test_framework REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

# creates the executable
add_executable(Net_test Net_test.cpp)
# indicates the include paths
target_link_libraries(Net_test Boost::unit_test_framework)

message(STATUS "BOOST_INCLUDE_DIRS " ${Boost_INCLUDE_DIRS})

# declares a test with our executable
add_test(NAME Net_test COMMAND Net_test)

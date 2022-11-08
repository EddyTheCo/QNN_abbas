cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
include(local_conf.cmake OPTIONAL)
project(QNN_module VERSION 0.1 DESCRIPTION "a Quantum neural network model ML-torch module for Fisher Matrix calculations" LANGUAGES CXX)

find_package(Torch REQUIRED)

include(FetchContent)
add_library(QNN_abbas include/custom-modules/qnn_abbas.hpp qnn_abbas.cpp)

if(USE_YAML)
	target_compile_definitions(QNN_abbas PUBLIC USE_YAML)
	if(NOT TARGET yaml-cpp)
		find_package(YAML-CPP CONFIG QUIET)
		if(NOT YAML-CPP_FOUND)
			message(STATUS "YAML-CPP was not found. Downloading")

			FetchContent_Declare(
				yaml-cpp
				GIT_REPOSITORY git@github.com:jbeder/yaml-cpp.git
				GIT_TAG yaml-cpp-0.7.0
				)
			FetchContent_MakeAvailable(yaml-cpp)
			install(TARGETS yaml-cpp EXPORT QNN_abbasTargets DESTINATION lib)
		else()
			message(STATUS "YAML FOUND in ${YAML-CPP_DIR}")
		endif(NOT YAML-CPP_FOUND)
	endif(NOT TARGET yaml-cpp)
	target_link_libraries(QNN_abbas PRIVATE  yaml-cpp)
endif(USE_YAML)

find_package(TBB CONFIG REQUIRED)
find_package(oneDPL CONFIG REQUIRED)

message(STATUS "Downloading qulacs")
FetchContent_Declare(
	qulacs
	GIT_REPOSITORY git@github.com:EddyTheCo/qulacs.git
	GIT_TAG main
	)
FetchContent_MakeAvailable(qulacs)
target_link_libraries(QNN_abbas PRIVATE  cppsim)

target_link_libraries(QNN_abbas  PUBLIC "${TORCH_LIBRARIES}")


target_link_libraries(QNN_abbas PRIVATE  ${TBB_IMPORTED_TARGETS} oneDPL)

target_include_directories(QNN_abbas PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	"$<INSTALL_INTERFACE:include>")

install(TARGETS QNN_abbas EXPORT QNN_abbasTargets DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/)

install(EXPORT QNN_abbasTargets
	FILE QNN_abbasTargets.cmake
	DESTINATION lib/cmake/QNN_abbas
	)
include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/QNN_abbasConfig.cmake"
	INSTALL_DESTINATION "lib/cmake/QNN_abbas"
	NO_SET_AND_CHECK_MACRO
	NO_CHECK_REQUIRED_COMPONENTS_MACRO
	)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/QNN_abbasConfigVersion.cmake"
	VERSION "0.1.1.0"
	COMPATIBILITY AnyNewerVersion
	)
install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/QNN_abbasConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/QNN_abbasConfigVersion.cmake
	DESTINATION lib/cmake/QNN_abbas
	)
export(EXPORT QNN_abbasTargets
	FILE "${CMAKE_CURRENT_BINARY_DIR}/QNN_abbasTargets.cmake"
	)
execute_process(COMMAND "git" "log" "--format='%H'" "-n 1"
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
	OUTPUT_VARIABLE  commit)
execute_process(COMMAND "git" "rev-parse" "--abbrev-ref" "HEAD"
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
	OUTPUT_VARIABLE  branch)

FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt  ${PROJECT_NAME}:\n)
FILE(APPEND ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt  \t commit=${commit})
FILE(APPEND ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt  \t branch=${branch}\n)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt DESTINATION .conf)

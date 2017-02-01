include_directories(${CMAKE_SOURCE_DIR}/src/)

add_executable(imgTimeMeasuring imgTimeMeasuring.cpp)
target_link_libraries(imgTimeMeasuring ImageData)

add_executable(utilDrawPolygons polygonDrawing.cpp)
target_link_libraries(utilDrawPolygons CellMap)

add_executable(utilDrawRegions regionDrawing.cpp)
target_link_libraries(utilDrawRegions CellMap)
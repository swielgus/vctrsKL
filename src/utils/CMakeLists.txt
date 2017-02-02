include_directories(${CMAKE_SOURCE_DIR}/src/)

add_executable(utilImgTimeMeasuring imgTimeMeasuring.cpp)
target_link_libraries(utilImgTimeMeasuring PolygonSideMap)

add_executable(utilDrawPolygons polygonDrawing.cpp)
target_link_libraries(utilDrawPolygons CellMap)

add_executable(utilDrawRegions regionDrawing.cpp)
target_link_libraries(utilDrawRegions CellMap)
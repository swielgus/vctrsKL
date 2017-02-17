pkg_search_module(PC_POPT QUIET popt)

FIND_PATH(POPT_INCLUDE_DIR
        NAMES popt.h
        HINTS
        ${PC_POPT_INCLUDEDIR}
        ${PC_POPT_INCLUDE_DIRS}
        )

FIND_LIBRARY(POPT_LIBRARY
        NAMES popt
        HINTS
        ${PC_POPT_LIBRARYDIR}
        ${PC_POPT_LIBRARY_DIRS}
        )

include_directories(${CMAKE_SOURCE_DIR}/src/ ${POPT_INCLUDE_DIR})

add_executable(utilImgTimeMeasuring imgTimeMeasuring.cpp)
target_link_libraries(utilImgTimeMeasuring CurveOptimizer ImageColorizer)

add_executable(utilDrawRegions regionDrawing.cpp)
target_link_libraries(utilDrawRegions PolygonSideMap)

add_executable(utilDrawPrototypeCurves prototypeCurveDrawing.cpp)
target_link_libraries(utilDrawPrototypeCurves CurveOptimizer ImageColorizer ${POPT_LIBRARY})
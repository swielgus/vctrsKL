find_package(PNG REQUIRED)
set(png++_FIND_REQUIRED TRUE)
find_path(
        png++_INCLUDE_DIRS
        NAMES png++/color.hpp  png++/config.hpp  png++/consumer.hpp  png++/convert_color_space.hpp
        png++/end_info.hpp  png++/error.hpp  png++/ga_pixel.hpp  png++/generator.hpp
        png++/gray_pixel.hpp  png++/image.hpp  png++/image_info.hpp  png++/index_pixel.hpp
        png++/info.hpp  png++/info_base.hpp  png++/io_base.hpp  png++/packed_pixel.hpp
        png++/palette.hpp  png++/pixel_buffer.hpp  png++/pixel_traits.hpp  png++/png.hpp
        png++/reader.hpp  png++/require_color_space.hpp  png++/rgb_pixel.hpp  png++/rgba_pixel.hpp
        png++/streaming_base.hpp png++/tRNS.hpp png++/types.hpp png++/writer.hpp
)
find_package_handle_standard_args(
        png++
        DEFAULT_MSG
        png++_INCLUDE_DIRS
)
include_directories(${PNG_INCLUDE_DIR} ${png++_INCLUDE_DIRS})

cuda_add_library(ImageData SHARED ImageData.cu )
target_link_libraries(ImageData ${PNG_LIBRARY})

cuda_add_library(PixelGraph SHARED PixelGraph.cu)
target_link_libraries(PixelGraph ImageData)

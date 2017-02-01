#include <iostream>
#include <chrono>
#include "ImageData.hpp"

int main()
{
    std::string fileName = "/home/sw/studia2016-2017Z/pracaMagisterska/conv/ftl.png";

    auto start = std::chrono::steady_clock::now();
    ImageData testedImage(fileName);
    auto duration = std::chrono::steady_clock::now() - start;
    std::cout << "\nTime measured: " << std::chrono::duration_cast< std::chrono::microseconds >(duration).count() << " microseconds \n"
              << "Time measured: " << std::chrono::duration_cast< std::chrono::milliseconds >(duration).count() << " milliseconds \n"
              << "Time measured: " << std::chrono::duration_cast< std::chrono::seconds >(duration).count() << " seconds \n";
    return 0;
}
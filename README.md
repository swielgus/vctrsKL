# vctrsKL

A C++ program that converts \*.png pixel art image to \*.svg vector representation using **[Kopf-Lischinski][1]** algorithm enhanced by GPU parallel operations provided by NVIDIA's [CUDA][2].

## Requirements
* CMake
* [CUDA][2] + compatible GPU (tested with version 8.0 on a device with compute capability 3.5)
* [Clipper][4]
* [LodePNG][3] (already included in this repository)
* [Google Test][5] (optional - for unit tests)

## Compiling
1. Generate files with CMake
1. Build with `make vctrsKL`

## Usage
```vctrsKL inputImage.png [OPTIONS]```

Run with `-?` or `--help` for descriptions of all optional parameters.

## Notes & possible improvements
* Thanks to parallel GPU computations, this implementation is able to process a 256x232 px image in 3 seconds, while *[libdepixelize][6]* needs 25+ minutes. It's still not even close to being comfortable for emulators. However, real-time implementation (about 8ms per frame) is possible - you can read about it [here][8].
* T-junctions are currently connected by straight lines so there is no smoothing like the methods proposed in the [original paper][1]. The output image may turn out to be distorted at these spots. Try running the program multiple times (optimized positions are randomized) or use the `-j` flag for (maybe) better outcomes.
* Secondary colors are rendered using SVG-supported gaussian blur for EACH of the original's pixels. This probably will make the output file very slow to open (I recommend Inkscape or a web browser). You might want to convert it back to a raster format or use `-g` flag to disable the functionality (which may result in color loss). Implementing a curve-region-for-every-color approach like [libdepixelize][6] might be a good idea.

## References
* *[Depixelizing pixel art][1]* (original paper + supplementary material with image examples)
* *[Connected Component Labeling in CUDA][7]* (method of identifying color groups which was used in this implementation)
* *[libdepixelize][6]* (popular implementation included in Inkscape 0.91+ vector graphics editor)
* *[Depixelizing pixel art in real-time][8]*

[1]: http://johanneskopf.de/publications/pixelart/
[2]: http://www.nvidia.com/cuda
[3]: http://lodev.org/lodepng/
[4]: http://www.angusj.com/delphi/clipper.php
[5]: https://github.com/google/googletest
[6]: https://launchpad.net/libdepixelize
[7]: http://hpcg.purdue.edu/papers/Stava2011CCL.pdf
[8]: https://www.cg.tuwien.ac.at/research/publications/2015/KREUZER-2015-DPA/
# Image Stitching

Image Stitching algorithm with multi-panoramas, gain compensation, simple blending, and multi-band blending. Part of a project for the Computer Vision course of CentraleSupélec.

The implementation is strongly based on the 2007 paper **Automatic Panoramic Image Stitching using Invariant Features** by Matthew Brown and David G. Lowe : <http://matthewalunbrown.com/papers/ijcv2007.pdf>

## Usage

Simply put all your images in a single folder (possibly images of different panoramas), and run the following command:

    python main.py /path/to/folder

For more options, see the command line help:

    python main.py --help

You will find sample images in the folder `samples/mountain` for you to try the algorithm.

## Examples

The below panoramas have been obtained with the default parameters.

![car panorama](samples/panoramas/car.jpg?raw=true)

![mountain panorama](samples/panoramas/mountain.jpg?raw=true)

![centralesupelec panorama](samples/panoramas/centralesupelec.jpg?raw=true)

## Detailed description

The process contains multiple steps to obtain the best-looking panorama as possible. Namely, the following steps are performed:

- **Feature detection**: the algorithm uses the SIFT algorithm to detect features in the images.
- **Feature matching**: the algorithm matches the features between the images.
- **Connected components**: the algorithm groups the images into connected components, each corresponding to a different panorama.
- **Panorama stitching**: the algorithm stitches the images together.
- **Gain compensation**: the algorithm compensates the gain of the images.
- **Blending**: the algorithm blends the images together. Two different blending methods are available: simple blending and multi-band blending. The simple blending is the one used by default, as the multi-band blending is slower and gives more blurred results.

![mountain panorama after panorama stitching](samples/steps/mountain_no_compensation.jpg?raw=true)
_Mountain panorama after panorama stitching without any compensation_

![mountain panorama after gain compensation](samples/steps/mountain_gain_compensation.jpg?raw=true)
_Mountain panorama after gain compensation_

![mountain panorama with multi-band blending](samples/steps/mountain_multi_band_blending.jpg?raw=true)
_Mountain panorama with multi-band blending_

![mountain panorama after panorama stitching](samples/panoramas/mountain.jpg?raw=true)
_Mountain panorama with simple blending_

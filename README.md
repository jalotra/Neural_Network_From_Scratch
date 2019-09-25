# Image-Processing-Package

The main objective of this module is to act as a starting package where any-body can implement and test out custom
algorithms using C language wrapped with Python3.

## Know the Image Data-Structure

Prior to know the structure of the package. Let's discuss how the image is stored in memory. The image is stored
in memory in form of a struct with four data-members.
```
struct{
	int w;
	int h;
	int c;
	float *data;
}
```
Here w represents the number of columns, h denotes number of rows and c denotes number of channels.
* data contains a array of pixel values stored in the CHW format. 

## Getting Started nad Installing

Things that you would require are : 
1. Gcc compiler
2. make
3. Python3  // Tested with python 3.6

I have myself included three custom methods namely 
1. get_pixel 
2. set_pixel
3. copy_image

You can read and learn more about these functions in the ./src/process_image.c file.

Now coming to the point. You have to run this single command to make the whole package.
```
make
```
## Implementing Your Custom methods

You will have to code out your custom functions in src folder in custom_filename.c. Plus then you have to include the object created in Makefile. And then you have to append the uwimg.py file so that you can use the method with python3. Checkout ctypes for more further documentation on this. 


## Testing Package

I have written included custom functions in uwimg.py.You can test out if your module works correctly or not just by executing the uwimg.py file.
Run
```
python3 uwimg.py 
```
And see the test photos in the test_photos folder. 



## Authors

* **Shivam Jalotra** - (https://shivamjalotra.me)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Everything is copied of from Pjreddie. I have just made use of his excellence in creating a great course for free namely Ancient Secrets of Computer Vision.

* My main motivation to create this file was to help budding image-processing geeks don't have any difficulty in setting up the environment to implement the algorithms that they have learn't very quickly.

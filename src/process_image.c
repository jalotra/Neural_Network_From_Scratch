#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"

// Returns the value at location (x,y,c) in an image
float get_pixel(image im, int x, int y, int c)
{
	// Overflow Protection 
	if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= im.w) x = im.w - 1; 
    if (y >= im.h) y = im.h - 1;

	int pixel_location = x + y*im.w + c*im.w*im.h;

	return im.data[pixel_location];
}

// Sets the value of the pixel in an image
void set_pixel(image im, int x, int y, int c, float v)
{
	// Overflows at edge cases
	if (x < 0)
        return;
    if (y < 0)
        return;
    if (x >= im.w)
        return;
    if (y >= im.h)
        return;

    int pixel_location = x + y*im.w + c*im.h*im.w;
    float *pixel_value = &im.data[pixel_location];
    // Set the value to be v
    *pixel_value = v; 
}

// Copies a image into another image and return the new image
image copy_image(image im)
{	
	image copy = make_image(im.w, im.h, im.c);
    memcpy(copy.data, im.data, im.w * im.h * im.c * sizeof(float));
    return copy;
}
  
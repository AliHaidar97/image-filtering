#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <mpi.h>
#include "gif_lib.h"

/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0

/* Represent one pixel from the image */
typedef struct pixel
{
    int r; /* Red */
    int g; /* Green */
    int b; /* Blue */
} pixel;

/* Represent one GIF image (animated or not */
typedef struct animated_gif
{
    int n_images;   /* Number of images */
    int *width;     /* Width of each image */
    int *height;    /* Height of each image */
    pixel **p;      /* Pixels of each image */
    GifFileType *g; /* Internal representation.
                       DO NOT MODIFY */
} animated_gif;

/*
 * Load a GIF image from a file and return a
 * structure of type animated_gif.
 */
animated_gif *
load_pixels(char *filename);

int output_modified_read_gif(char *filename, GifFileType *g);

int store_pixels(char *filename, animated_gif *image);
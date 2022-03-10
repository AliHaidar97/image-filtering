/*
 * INF560
 *
 * Image Filtering Project
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

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
    int n_images; /* Number of images */
    int* width; /* Width of each image */
    int* height; /* Height of each image */
    pixel** p; /* Pixels of each image */
    GifFileType* g; /* Internal representation.
                         DO NOT MODIFY */
} animated_gif;

/*
 * Load a GIF image from a file and return a
 * structure of type animated_gif.
 */
animated_gif*
load_pixels(char* filename)
{
    GifFileType* g;
    ColorMapObject* colmap;
    int error;
    int n_images;
    int* width;
    int* height;
    pixel** p;
    int i;
    animated_gif* image;

    /* Open the GIF image (read mode) */
    g = DGifOpenFileName(filename, &error);
    if (g == NULL)
    {
        fprintf(stderr, "Error DGifOpenFileName %s\n", filename);
        return NULL;
    }

    /* Read the GIF image */
    error = DGifSlurp(g);
    if (error != GIF_OK)
    {
        fprintf(stderr,
            "Error DGifSlurp: %d <%s>\n", error, GifErrorString(g->Error));
        return NULL;
    }

    /* Grab the number of images and the size of each image */
    n_images = g->ImageCount;

    width = (int*)malloc(n_images * sizeof(int));
    if (width == NULL)
    {
        fprintf(stderr, "Unable to allocate width of size %d\n",
            n_images);
        return 0;
    }

    height = (int*)malloc(n_images * sizeof(int));
    if (height == NULL)
    {
        fprintf(stderr, "Unable to allocate height of size %d\n",
            n_images);
        return 0;
    }

    /* Fill the width and height */
    for (i = 0; i < n_images; i++)
    {
        width[i] = g->SavedImages[i].ImageDesc.Width;
        height[i] = g->SavedImages[i].ImageDesc.Height;

#if SOBELF_DEBUG
        printf("Image %d: l:%d t:%d w:%d h:%d interlace:%d localCM:%p\n",
            i,
            g->SavedImages[i].ImageDesc.Left,
            g->SavedImages[i].ImageDesc.Top,
            g->SavedImages[i].ImageDesc.Width,
            g->SavedImages[i].ImageDesc.Height,
            g->SavedImages[i].ImageDesc.Interlace,
            g->SavedImages[i].ImageDesc.ColorMap
        );
#endif
    }


    /* Get the global colormap */
    colmap = g->SColorMap;
    if (colmap == NULL)
    {
        fprintf(stderr, "Error global colormap is NULL\n");
        return NULL;
    }

#if SOBELF_DEBUG
    printf("Global color map: count:%d bpp:%d sort:%d\n",
        g->SColorMap->ColorCount,
        g->SColorMap->BitsPerPixel,
        g->SColorMap->SortFlag
    );
#endif

    /* Allocate the array of pixels to be returned */
    p = (pixel**)malloc(n_images * sizeof(pixel*));
    if (p == NULL)
    {
        fprintf(stderr, "Unable to allocate array of %d images\n",
            n_images);
        return NULL;
    }

    for (i = 0; i < n_images; i++)
    {
        p[i] = (pixel*)malloc(width[i] * height[i] * sizeof(pixel));
        if (p[i] == NULL)
        {
            fprintf(stderr, "Unable to allocate %d-th array of %d pixels\n",
                i, width[i] * height[i]);
            return NULL;
        }
    }

    /* Fill pixels */

    /* For each image */
    for (i = 0; i < n_images; i++)
    {
        int j;

        /* Get the local colormap if needed */
        if (g->SavedImages[i].ImageDesc.ColorMap)
        {

            /* TODO No support for local color map */
            fprintf(stderr, "Error: application does not support local colormap\n");
            return NULL;

            colmap = g->SavedImages[i].ImageDesc.ColorMap;
        }

        /* Traverse the image and fill pixels */
        for (j = 0; j < width[i] * height[i]; j++)
        {
            int c;

            c = g->SavedImages[i].RasterBits[j];

            p[i][j].r = colmap->Colors[c].Red;
            p[i][j].g = colmap->Colors[c].Green;
            p[i][j].b = colmap->Colors[c].Blue;
        }
    }

    /* Allocate image info */
    image = (animated_gif*)malloc(sizeof(animated_gif));
    if (image == NULL)
    {
        fprintf(stderr, "Unable to allocate memory for animated_gif\n");
        return NULL;
    }

    /* Fill image fields */
    image->n_images = n_images;
    image->width = width;
    image->height = height;
    image->p = p;
    image->g = g;

#if SOBELF_DEBUG
    printf("-> GIF w/ %d image(s) with first image of size %d x %d\n",
        image->n_images, image->width[0], image->height[0]);
#endif

    return image;
}

int
output_modified_read_gif(char* filename, GifFileType* g)
{
    GifFileType* g2;
    int error2;

#if SOBELF_DEBUG
    printf("Starting output to file %s\n", filename);
#endif

    g2 = EGifOpenFileName(filename, false, &error2);
    if (g2 == NULL)
    {
        fprintf(stderr, "Error EGifOpenFileName %s\n",
            filename);
        return 0;
    }

    g2->SWidth = g->SWidth;
    g2->SHeight = g->SHeight;
    g2->SColorResolution = g->SColorResolution;
    g2->SBackGroundColor = g->SBackGroundColor;
    g2->AspectByte = g->AspectByte;
    g2->SColorMap = g->SColorMap;
    g2->ImageCount = g->ImageCount;
    g2->SavedImages = g->SavedImages;
    g2->ExtensionBlockCount = g->ExtensionBlockCount;
    g2->ExtensionBlocks = g->ExtensionBlocks;

    error2 = EGifSpew(g2);
    if (error2 != GIF_OK)
    {
        fprintf(stderr, "Error after writing g2: %d <%s>\n",
            error2, GifErrorString(g2->Error));
        return 0;
    }

    return 1;
}


int
store_pixels(char* filename, animated_gif* image)
{
    int n_colors = 0;
    pixel** p;
    int i, j, k;
    GifColorType* colormap;

    /* Initialize the new set of colors */
    colormap = (GifColorType*)malloc(256 * sizeof(GifColorType));
    if (colormap == NULL)
    {
        fprintf(stderr,
            "Unable to allocate 256 colors\n");
        return 0;
    }

    /* Everything is white by default */
    for (i = 0; i < 256; i++)
    {
        colormap[i].Red = 255;
        colormap[i].Green = 255;
        colormap[i].Blue = 255;
    }

    /* Change the background color and store it */
    int moy;
    moy = (
        image->g->SColorMap->Colors[image->g->SBackGroundColor].Red
        +
        image->g->SColorMap->Colors[image->g->SBackGroundColor].Green
        +
        image->g->SColorMap->Colors[image->g->SBackGroundColor].Blue
        ) / 3;
    if (moy < 0) moy = 0;
    if (moy > 255) moy = 255;

#if SOBELF_DEBUG
    printf("[DEBUG] Background color (%d,%d,%d) -> (%d,%d,%d)\n",
        image->g->SColorMap->Colors[image->g->SBackGroundColor].Red,
        image->g->SColorMap->Colors[image->g->SBackGroundColor].Green,
        image->g->SColorMap->Colors[image->g->SBackGroundColor].Blue,
        moy, moy, moy);
#endif

    colormap[0].Red = moy;
    colormap[0].Green = moy;
    colormap[0].Blue = moy;

    image->g->SBackGroundColor = 0;

    n_colors++;

    /* Process extension blocks in main structure */
    for (j = 0; j < image->g->ExtensionBlockCount; j++)
    {
        int f;

        f = image->g->ExtensionBlocks[j].Function;
        if (f == GRAPHICS_EXT_FUNC_CODE)
        {
            int tr_color = image->g->ExtensionBlocks[j].Bytes[3];

            if (tr_color >= 0 &&
                tr_color < 255)
            {

                int found = -1;

                moy =
                    (
                        image->g->SColorMap->Colors[tr_color].Red
                        +
                        image->g->SColorMap->Colors[tr_color].Green
                        +
                        image->g->SColorMap->Colors[tr_color].Blue
                        ) / 3;
                if (moy < 0) moy = 0;
                if (moy > 255) moy = 255;

#if SOBELF_DEBUG
                printf("[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
                    i,
                    image->g->SColorMap->Colors[tr_color].Red,
                    image->g->SColorMap->Colors[tr_color].Green,
                    image->g->SColorMap->Colors[tr_color].Blue,
                    moy, moy, moy);
#endif

                for (k = 0; k < n_colors; k++)
                {
                    if (
                        moy == colormap[k].Red
                        &&
                        moy == colormap[k].Green
                        &&
                        moy == colormap[k].Blue
                        )
                    {
                        found = k;
                    }
                }
                if (found == -1)
                {
                    if (n_colors >= 256)
                    {
                        fprintf(stderr,
                            "Error: Found too many colors inside the image\n"
                        );
                        return 0;
                    }

#if SOBELF_DEBUG
                    printf("[DEBUG]\tNew color %d\n",
                        n_colors);
#endif

                    colormap[n_colors].Red = moy;
                    colormap[n_colors].Green = moy;
                    colormap[n_colors].Blue = moy;


                    image->g->ExtensionBlocks[j].Bytes[3] = n_colors;

                    n_colors++;
                }
                else
                {
#if SOBELF_DEBUG
                    printf("[DEBUG]\tFound existing color %d\n",
                        found);
#endif
                    image->g->ExtensionBlocks[j].Bytes[3] = found;
                }
            }
        }
    }

    for (i = 0; i < image->n_images; i++)
    {
        for (j = 0; j < image->g->SavedImages[i].ExtensionBlockCount; j++)
        {
            int f;

            f = image->g->SavedImages[i].ExtensionBlocks[j].Function;
            if (f == GRAPHICS_EXT_FUNC_CODE)
            {
                int tr_color = image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3];

                if (tr_color >= 0 &&
                    tr_color < 255)
                {

                    int found = -1;

                    moy =
                        (
                            image->g->SColorMap->Colors[tr_color].Red
                            +
                            image->g->SColorMap->Colors[tr_color].Green
                            +
                            image->g->SColorMap->Colors[tr_color].Blue
                            ) / 3;
                    if (moy < 0) moy = 0;
                    if (moy > 255) moy = 255;

#if SOBELF_DEBUG
                    printf("[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
                        i,
                        image->g->SColorMap->Colors[tr_color].Red,
                        image->g->SColorMap->Colors[tr_color].Green,
                        image->g->SColorMap->Colors[tr_color].Blue,
                        moy, moy, moy);
#endif

                    for (k = 0; k < n_colors; k++)
                    {
                        if (
                            moy == colormap[k].Red
                            &&
                            moy == colormap[k].Green
                            &&
                            moy == colormap[k].Blue
                            )
                        {
                            found = k;
                        }
                    }
                    if (found == -1)
                    {
                        if (n_colors >= 256)
                        {
                            fprintf(stderr,
                                "Error: Found too many colors inside the image\n"
                            );
                            return 0;
                        }

#if SOBELF_DEBUG
                        printf("[DEBUG]\tNew color %d\n",
                            n_colors);
#endif

                        colormap[n_colors].Red = moy;
                        colormap[n_colors].Green = moy;
                        colormap[n_colors].Blue = moy;


                        image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = n_colors;

                        n_colors++;
                    }
                    else
                    {
#if SOBELF_DEBUG
                        printf("[DEBUG]\tFound existing color %d\n",
                            found);
#endif
                        image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = found;
                    }
                }
            }
        }
    }

#if SOBELF_DEBUG
    printf("[DEBUG] Number of colors after background and transparency: %d\n",
        n_colors);
#endif

    p = image->p;

    /* Find the number of colors inside the image */
    for (i = 0; i < image->n_images; i++)
    {

#if SOBELF_DEBUG
        printf("OUTPUT: Processing image %d (total of %d images) -> %d x %d\n",
            i, image->n_images, image->width[i], image->height[i]);
#endif

        for (j = 0; j < image->width[i] * image->height[i]; j++)
        {
            int found = 0;
            for (k = 0; k < n_colors; k++)
            {
                if (p[i][j].r == colormap[k].Red &&
                    p[i][j].g == colormap[k].Green &&
                    p[i][j].b == colormap[k].Blue)
                {
                    found = 1;
                }
            }

            if (found == 0)
            {
                if (n_colors >= 256)
                {
                    fprintf(stderr,
                        "Error: Found too many colors inside the image\n"
                    );
                    return 0;
                }

#if SOBELF_DEBUG
                printf("[DEBUG] Found new %d color (%d,%d,%d)\n",
                    n_colors, p[i][j].r, p[i][j].g, p[i][j].b);
#endif

                colormap[n_colors].Red = p[i][j].r;
                colormap[n_colors].Green = p[i][j].g;
                colormap[n_colors].Blue = p[i][j].b;
                n_colors++;
            }
        }
    }

#if SOBELF_DEBUG
    printf("OUTPUT: found %d color(s)\n", n_colors);
#endif


    /* Round up to a power of 2 */
    if (n_colors != (1 << GifBitSize(n_colors)))
    {
        n_colors = (1 << GifBitSize(n_colors));
    }

#if SOBELF_DEBUG
    printf("OUTPUT: Rounding up to %d color(s)\n", n_colors);
#endif

    /* Change the color map inside the animated gif */
    ColorMapObject* cmo;

    cmo = GifMakeMapObject(n_colors, colormap);
    if (cmo == NULL)
    {
        fprintf(stderr, "Error while creating a ColorMapObject w/ %d color(s)\n",
            n_colors);
        return 0;
    }

    image->g->SColorMap = cmo;

    /* Update the raster bits according to color map */
    for (i = 0; i < image->n_images; i++)
    {
        for (j = 0; j < image->width[i] * image->height[i]; j++)
        {
            int found_index = -1;
            for (k = 0; k < n_colors; k++)
            {
                if (p[i][j].r == image->g->SColorMap->Colors[k].Red &&
                    p[i][j].g == image->g->SColorMap->Colors[k].Green &&
                    p[i][j].b == image->g->SColorMap->Colors[k].Blue)
                {
                    found_index = k;
                }
            }

            if (found_index == -1)
            {
                fprintf(stderr,
                    "Error: Unable to find a pixel in the color map\n");
                return 0;
            }

            image->g->SavedImages[i].RasterBits[j] = found_index;
        }
    }


    /* Write the final image */
    if (!output_modified_read_gif(filename, image->g)) { return 0; }

    return 1;
}

void
apply_gray_filter(animated_gif* image)
{
    int i, j;
    pixel** p;

    p = image->p;

    for (i = 0; i < image->n_images; i++)
    {
        for (j = 0; j < image->width[i] * image->height[i]; j++)
        {
            int moy;

            moy = (p[i][j].r + p[i][j].g + p[i][j].b) / 3;
            if (moy < 0) moy = 0;
            if (moy > 255) moy = 255;

            p[i][j].r = moy;
            p[i][j].g = moy;
            p[i][j].b = moy;
        }
    }
}

#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)

void apply_gray_line(animated_gif* image)
{
    int i, j, k;
    pixel** p;

    p = image->p;

    for (i = 0; i < image->n_images; i++)
    {
        for (j = 0; j < 10; j++)
        {
            for (k = image->width[i] / 2; k < image->width[i]; k++)
            {
                p[i][CONV(j, k, image->width[i])].r = 0;
                p[i][CONV(j, k, image->width[i])].g = 0;
                p[i][CONV(j, k, image->width[i])].b = 0;
            }
        }
    }
}

void
apply_blur_filter(animated_gif* image, int size, int threshold)
{
    int i, j, k;
    int width, height;
    int end = 0;
    int n_iter = 0;

    pixel** p;
    pixel* new;

    /* Get the pixels of all images */
    p = image->p;


    /* Process all images */
    for (i = 0; i < image->n_images; i++)
    {
        n_iter = 0;
        width = image->width[i];
        height = image->height[i];

        /* Allocate array of new pixels */
        new = (pixel*)malloc(width * height * sizeof(pixel));


        /* Perform at least one blur iteration */
        do
        {
            end = 1;
            n_iter++;


            for (j = 0; j < height - 1; j++)
            {
                for (k = 0; k < width - 1; k++)
                {
                    new[CONV(j, k, width)].r = p[i][CONV(j, k, width)].r;
                    new[CONV(j, k, width)].g = p[i][CONV(j, k, width)].g;
                    new[CONV(j, k, width)].b = p[i][CONV(j, k, width)].b;
                }
            }

          

            /* Apply blur on top part of image (10%) */
            for (j = size; j < height / 10 - size; j++)
            {
                for (k = size; k < width - size; k++)
                {
                    int stencil_j, stencil_k;
                    int t_r = 0;
                    int t_g = 0;
                    int t_b = 0;

                    for (stencil_j = -size; stencil_j <= size; stencil_j++)
                    {
                        for (stencil_k = -size; stencil_k <= size; stencil_k++)
                        {
                            t_r += p[i][CONV(j + stencil_j, k + stencil_k, width)].r;
                            t_g += p[i][CONV(j + stencil_j, k + stencil_k, width)].g;
                            t_b += p[i][CONV(j + stencil_j, k + stencil_k, width)].b;
                        }
                    }

                    new[CONV(j, k, width)].r = t_r / ((2 * size + 1) * (2 * size + 1));
                    new[CONV(j, k, width)].g = t_g / ((2 * size + 1) * (2 * size + 1));
                    new[CONV(j, k, width)].b = t_b / ((2 * size + 1) * (2 * size + 1));
                }
            }

            /* Copy the middle part of the image */
            for (j = height / 10 - size; j < height * 0.9 + size; j++)
            {
                for (k = size; k < width - size; k++)
                {
                    new[CONV(j, k, width)].r = p[i][CONV(j, k, width)].r;
                    new[CONV(j, k, width)].g = p[i][CONV(j, k, width)].g;
                    new[CONV(j, k, width)].b = p[i][CONV(j, k, width)].b;
                }
            }

            /* Apply blur on the bottom part of the image (10%) */
            for (j = height * 0.9 + size; j < height - size; j++)
            {
                for (k = size; k < width - size; k++)
                {
                    int stencil_j, stencil_k;
                    int t_r = 0;
                    int t_g = 0;
                    int t_b = 0;

                    for (stencil_j = -size; stencil_j <= size; stencil_j++)
                    {
                        for (stencil_k = -size; stencil_k <= size; stencil_k++)
                        {
                            t_r += p[i][CONV(j + stencil_j, k + stencil_k, width)].r;
                            t_g += p[i][CONV(j + stencil_j, k + stencil_k, width)].g;
                            t_b += p[i][CONV(j + stencil_j, k + stencil_k, width)].b;
                        }
                    }

                    new[CONV(j, k, width)].r = t_r / ((2 * size + 1) * (2 * size + 1));
                    new[CONV(j, k, width)].g = t_g / ((2 * size + 1) * (2 * size + 1));
                    new[CONV(j, k, width)].b = t_b / ((2 * size + 1) * (2 * size + 1));
                }
            }

            for (j = 1; j < height - 1; j++)
            {
                for (k = 1; k < width - 1; k++)
                {

                    float diff_r;
                    float diff_g;
                    float diff_b;

                    diff_r = (new[CONV(j, k, width)].r - p[i][CONV(j, k, width)].r);
                    diff_g = (new[CONV(j, k, width)].g - p[i][CONV(j, k, width)].g);
                    diff_b = (new[CONV(j, k, width)].b - p[i][CONV(j, k, width)].b);

                    if (diff_r > threshold || -diff_r > threshold
                        ||
                        diff_g > threshold || -diff_g > threshold
                        ||
                        diff_b > threshold || -diff_b > threshold
                        ) {
                        end = 0;
                    }

                    p[i][CONV(j, k, width)].r = new[CONV(j, k, width)].r;
                    p[i][CONV(j, k, width)].g = new[CONV(j, k, width)].g;
                    p[i][CONV(j, k, width)].b = new[CONV(j, k, width)].b;
                }
            }

        }         while (threshold > 0 && !end);

#if SOBELF_DEBUG
        printf("BLUR: number of iterations for image %d\n", n_iter);
#endif

        free(new);
    }

}

void
apply_sobel_filter(animated_gif* image)
{
    int i, j, k;
    int width, height;

    pixel** p;

    p = image->p;

    for (i = 0; i < image->n_images; i++)
    {
        width = image->width[i];
        height = image->height[i];

        pixel* sobel;

        sobel = (pixel*)malloc(width * height * sizeof(pixel));

        for (j = 1; j < height - 1; j++)
        {
            for (k = 1; k < width - 1; k++)
            {
                int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
                int pixel_blue_so, pixel_blue_s, pixel_blue_se;
                int pixel_blue_o, pixel_blue, pixel_blue_e;

                float deltaX_blue;
                float deltaY_blue;
                float val_blue;

                pixel_blue_no = p[i][CONV(j - 1, k - 1, width)].b;
                pixel_blue_n = p[i][CONV(j - 1, k, width)].b;
                pixel_blue_ne = p[i][CONV(j - 1, k + 1, width)].b;
                pixel_blue_so = p[i][CONV(j + 1, k - 1, width)].b;
                pixel_blue_s = p[i][CONV(j + 1, k, width)].b;
                pixel_blue_se = p[i][CONV(j + 1, k + 1, width)].b;
                pixel_blue_o = p[i][CONV(j, k - 1, width)].b;
                pixel_blue = p[i][CONV(j, k, width)].b;
                pixel_blue_e = p[i][CONV(j, k + 1, width)].b;

                deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2 * pixel_blue_o + 2 * pixel_blue_e - pixel_blue_so + pixel_blue_se;

                deltaY_blue = pixel_blue_se + 2 * pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2 * pixel_blue_n - pixel_blue_no;

                val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue) / 4;


                if (val_blue > 50)
                {
                    sobel[CONV(j, k, width)].r = 255;
                    sobel[CONV(j, k, width)].g = 255;
                    sobel[CONV(j, k, width)].b = 255;
                }
                else
                {
                    sobel[CONV(j, k, width)].r = 0;
                    sobel[CONV(j, k, width)].g = 0;
                    sobel[CONV(j, k, width)].b = 0;
                }
            }
        }

        for (j = 1; j < height - 1; j++)
        {
            for (k = 1; k < width - 1; k++)
            {
                p[i][CONV(j, k, width)].r = sobel[CONV(j, k, width)].r;
                p[i][CONV(j, k, width)].g = sobel[CONV(j, k, width)].g;
                p[i][CONV(j, k, width)].b = sobel[CONV(j, k, width)].b;
            }
        }

        free(sobel);
    }

}

//CUDA PART


#define numThread 5 // 5 threads in a block
#define numBlock 6  // 6 blocks


//load the image to GPU
void loadImageToCuda(animated_gif* image, int index,  pixel* d_p) {

    //Memory allocation on cuda
    cudaMalloc((void**)&d_p,
        image->height[index] * image->width[index] * sizeof(pixel));
   
    cudaMemcpy(d_p, image->pixel[index],
        image->height[index] * image->width[index] * sizeof(pixel),
        cudaMemcpyHostToDevice);
}

//load the image from GPU back to CPU
void loadImageToHost(animated_gif* image, int index, pixel* d_p) {

    cudaMemcpy(image->pixel[index] , d_p,
        image->height[index] * image->width[index] * sizeof(pixel),
        cudaMemcpyDeviceToHost);

}


void freeDataFromCuda(pixel* d_p) {
    
    free(d_p);

}


/* Function to apply gray filter on Cuda*/
__global__ void apply_gray_filter_on_Cuda_kernel(pixel* d_p, int height, int width)
{

    int i;
    i = blockDim.x * blockIdx.x + threadIdx.x;
    int total = numThread * numBlock;
    while (i < height*width) {

        int moy;

        moy = (p[i].r + p[i].g + p[i].b) / 3;
        if (moy < 0) moy = 0;
        if (moy > 255) moy = 255;

        d_p[i].r = moy;
        d_p[i].g = moy;
        d_p[i].b = moy;


        i += total;
    } 

}




void apply_gray_filter_on_Cuda(animated_gif* image) {

    int i;
    
    for (i = 0; i < image->n_images; i++)
    {
        pixel* d_p;
        loadImageToCuda(image, i, d_p);
        apply_gray_filter_on_Cuda_kernel << <numBlock, numThread >> > (d_p, image->height[i], image->width[i]);
        loadImageToHost(image, d_p);
        freeDataFromCuda(d_p);
    }

}






__global__ void apply_sobel_filter_on_Cuda_kernel_first_phase(pixel* d_p, pixel* d_sobel, int height, int width)
{

    int i, j, k;
    i = blockDim.x * blockIdx.x + threadIdx.x;
    int total = numThread * numBlock;
    while (1) {

        j = i / (width - 1) + 1;
        k = i % (width - 1) + 1;

        if (j >= height - 1)break;
        if (k >= width - 1)break;


        int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
        int pixel_blue_so, pixel_blue_s, pixel_blue_se;
        int pixel_blue_o, pixel_blue, pixel_blue_e;

        float deltaX_blue;
        float deltaY_blue;
        float val_blue;

        pixel_blue_no = d_p[CONV(j - 1, k - 1, width)].b;
        pixel_blue_n = d_p[CONV(j - 1, k, width)].b;
        pixel_blue_ne = d_p[CONV(j - 1, k + 1, width)].b;
        pixel_blue_so = d_p[CONV(j + 1, k - 1, width)].b;
        pixel_blue_s = d_p[CONV(j + 1, k, width)].b;
        pixel_blue_se = d_p[CONV(j + 1, k + 1, width)].b;
        pixel_blue_o = d_p[CONV(j, k - 1, width)].b;
        pixel_blue = d_p[CONV(j, k, width)].b;
        pixel_blue_e = d_p[CONV(j, k + 1, width)].b;

        deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2 * pixel_blue_o + 2 * pixel_blue_e - pixel_blue_so + pixel_blue_se;

        deltaY_blue = pixel_blue_se + 2 * pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2 * pixel_blue_n - pixel_blue_no;

        val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue) / 4;


        if (val_blue > 50)
        {
            d_sobel[CONV(j, k, width)].r = 255;
            d_sobel[CONV(j, k, width)].g = 255;
            d_sobel[CONV(j, k, width)].b = 255;
        }
        else
        {
            d_sobel[CONV(j, k, width)].r = 0;
            d_sobel[CONV(j, k, width)].g = 0;
            d_sobel[CONV(j, k, width)].b = 0;

        }
        i += total;
    }
}



//applying the convolution part
__global__ void apply_conv_on_Cuda_kernel(pixel* d_p, pixel* d_temp_p,int l1,int r1,int l2,int r2)
{

    int i, j, k;
    i = blockDim.x * blockIdx.x + threadIdx.x;
    int total = numThread * numBlock;
    while (1) {
        

        j = i / (r2 - l2) + l1;
        k = i % (r2 -l2) + l2;

        if (j >= r1)break;
       

        d_p[CONV(j, k, width)].r = d_temp_p[CONV(j, k, width)].r;
        d_p[CONV(j, k, width)].g = d_temp_p[CONV(j, k, width)].g;
        d_p[CONV(j, k, width)].b = d_temp_p[CONV(j, k, width)].b;
           
      
        i += total;
    }
}



void apply_sobel_filter_on_Cuda(animated_gif* image) {

    int i;
    pixel* d_p, d_sobel;
    int height, width;
    for (i = 0; i < image->n_images; i++)
    {
        width = image->width[i];
        height = image->height[i];

        loadImageToCuda(image, i, d_p);
        loadImageToCuda(image, i, d_sobel);
        
        apply_sobel_filter_on_Cuda_kernel_first_phase << <numBlock, numThread >> > (d_p, d_sobel, height, width);
        apply_conv_on_Cuda_kernel<< <numBlock, numThread >> > (d_p, d_sobel, 1, height-1, 1, width-1);


        loadImageToHost(image, d_p);
        freeDataFromCuda(d_p);
        freeDataFromCuda(d_sobel);
    }


}



__global__ void apply_blur_filter_on_Cuda_kernel(pixel* d_p, pixel* d_new, int l1, int r1, int l2, int r2,int size,int height,int width)
{

    int i, j, k;
    i = blockDim.x * blockIdx.x + threadIdx.x;
    int total = numThread * numBlock;
    while (1) {

        j = i / (r2 - l2) + l1;
        k = i % (r2 - l2) + l2;

        if (j >= r1)break;
        
        int stencil_j, stencil_k;
        int t_r = 0;
        int t_g = 0;
        int t_b = 0;

        for (stencil_j = -size; stencil_j <= size; stencil_j++)
        {
            for (stencil_k = -size; stencil_k <= size; stencil_k++)
            {
                t_r += d_p[CONV(j + stencil_j, k + stencil_k, width)].r;
                t_g += d_p[CONV(j + stencil_j, k + stencil_k, width)].g;
                t_b += d_p][CONV(j + stencil_j, k + stencil_k, width)].b;
            }
        }

        d_new[CONV(j, k, width)].r = t_r / ((2 * size + 1) * (2 * size + 1));
        d_new[CONV(j, k, width)].g = t_g / ((2 * size + 1) * (2 * size + 1));
        d_new[CONV(j, k, width)].b = t_b / ((2 * size + 1) * (2 * size + 1));
      

        i += total;
    }
}









__global__ void check_blur_filter_on_Cuda_kernel(pixel* d_p, pixel* d_new,int * d_end, int l1, int r1, int l2, int r2, int size, int height, int width, int threshold)
{

    int i, j, k;
    i = blockDim.x * blockIdx.x + threadIdx.x;
    int total = numThread * numBlock;
    while (1) {

        j = i / (r2 - l2) + l1;
        k = i % (r2 - l2) + l2;

        if (j >= r1)break;

        float diff_r;
        float diff_g;
        float diff_b;

        diff_r = (d_new[CONV(j, k, width)].r - d_p[CONV(j, k, width)].r);
        diff_g = (d_new[CONV(j, k, width)].g - d_p[CONV(j, k, width)].g);
        diff_b = (d_new[CONV(j, k, width)].b - d_p[CONV(j, k, width)].b);

        if (diff_r > threshold || -diff_r > threshold
            ||
            diff_g > threshold || -diff_g > threshold
            ||
            diff_b > threshold || -diff_b > threshold
            ) {
            (*d_end) = 0;


        d_p[CONV(j, k, width)].r = d_new[CONV(j, k, width)].r;
        d_p[CONV(j, k, width)].g = d_new[CONV(j, k, width)].g;
        d_p[CONV(j, k, width)].b = d_new[CONV(j, k, width)].b;

        i += total;
    }
}







void apply_blur_filter_on_Cuda(animated_gif* image, int size, int threshold) {


    int i, j, k;
    int width, height;
    int* end = (int*)malloc(1 * sizeof(int));
    
    int n_iter = 0;
    pixel* d_p, d_new;
    int* d_end;
    cudaMalloc((void**)&d_end,
        1* sizeof(int));

    

    for (i = 0; i < image->n_images; i++)
    {
        n_iter = 0;
        (*end) = 1;
        width = image->width[i];
        height = image->height[i];
        
        

        
        /* Allocate array of new pixels */
        loadImageToCuda(image, i, d_p);
        loadImageToCuda(image, i, d_new);


        /* Perform at least one blur iteration */
        do
        {
            (*end) = 1;
            cudaMemcpy(d_end, end,
                1 * sizeof(int),
                cudaMemcpyHostToDevice);
            n_iter++;

            apply_conv_on_Cuda_kernel << <numBlock, numThread >> > (d_new, d_p, 0, width - 1, 0, height - 1);

            /* Apply blur on top part of image (10%) */
            apply_blur_filter_on_Cuda_kernel << <numBlock, numThread >> > (d_p, d_new, size, height / 10, size, width - size, size, height, width);
        

            /* Copy the middle part of the image */
            apply_conv_on_Cuda_kernel << <numBlock, numThread >> > (d_new, d_p, height / 10, height * 0.9 + size, size, width - size);
            

            /* Apply blur on the bottom part of the image (10%) */
            apply_blur_filter_on_Cuda_kernel << <numBlock, numThread >> > (d_p, d_new, height * 0.9 + size, height - size, size, width - size, size, height, width);
          
            /*Check blur filter*/
            check_blur_filter_on_Cuda_kernel << <numBlock, numThread >> > (d_p, d_new, d_end, 1, height - 1, 1, width, size, height, width, threshold);
            
            cudaMemcpy(end, d_end,
                1 * sizeof(int),
                cudaMemcpyDeviceToHost);

        } while (threshold > 0 && !(*end));

#if SOBELF_DEBUG
        printf("BLUR: number of iterations for image %d\n", n_iter);
#endif

        loadImageToHost(image, d_p);
        freeDataFromCuda(d_p);
        freeDataFromCuda(d_sobel);
        
    }
    free(d_end);
    free(end);

}









void image_filtering_Cuda(animated_gif* image) {
    
    /*Apply gray filter on pixels*/
    apply_gray_filter_on_Cuda(image);

    /* Apply blur filter with convergence value */
    apply_blur_filter_on_Cuda(image, 5, 20);

    /* Apply sobel filter on pixels */
    apply_sobel_filter_on_Cuda(image);


}





/*
 * Main entry point
 */
int
main(int argc, char** argv)
{
    char* input_filename;
    char* output_filename;
    animated_gif* image;
    struct timeval t1, t2;
    double duration;

    /* Check command-line arguments */
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s input.gif output.gif \n", argv[0]);
        return 1;
    }

    input_filename = argv[1];
    output_filename = argv[2];

    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Load file and store the pixels in array */
    image = load_pixels(input_filename);
    if (image == NULL) { return 1; }

    /* IMPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("GIF loaded from file %s with %d image(s) in %lf s\n",
        input_filename, image->n_images, duration);

    /* FILTER Timer start */
    gettimeofday(&t1, NULL);

    //filter the image using Cuda
    image_filtering_Cuda(image);

    /* FILTER Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("SOBEL done in %lf s\n", duration);

    /* EXPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Store file from array of pixels to GIF file */
    if (!store_pixels(output_filename, image)) { return 1; }

    /* EXPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("Export done in %lf s in file %s\n", duration, output_filename);

    return 0;
}

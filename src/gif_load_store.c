#include "gif_load_store.h"

/*
 * Load a GIF image from a file and return a
 * structure of type animated_gif.
 */
animated_gif *
load_pixels(char *filename)
{
    GifFileType *g;
    ColorMapObject *colmap;
    int error;
    int n_images;
    int *width;
    int *height;
    pgrey **p;
    int i;

    animated_gif *image;

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

    width = (int *)malloc(n_images * sizeof(int));
    if (width == NULL)
    {
        fprintf(stderr, "Unable to allocate width of size %d\n",
                n_images);
        return 0;
    }

    height = (int *)malloc(n_images * sizeof(int));
    if (height == NULL)
    {
        fprintf(stderr, "Unable to allocate height of size %d\n",
                n_images);
        return 0;
    }

    /* Fill the width and height */

#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
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
                   g->SavedImages[i].ImageDesc.ColorMap);
#endif
        }
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
           g->SColorMap->SortFlag);
#endif

    /* Allocate the array of pixels to be returned */
    p = (pgrey **)malloc(n_images * sizeof(pgrey *));
    if (p == NULL)
    {
        fprintf(stderr, "Unable to allocate array of %d images\n",
                n_images);
        return NULL;
    }

#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
        for (i = 0; i < n_images; i++)
        {
            p[i] = (pgrey *)malloc(width[i] * height[i] * sizeof(pgrey));
            /*
             if (p[i] == NULL)
             {
                 fprintf(stderr, "Unable to allocate %d-th array of %d pixels\n",
                     i, width[i] * height[i]);
                 return NULL;
             }
             */
        }
    }
    /* Fill pixels */

    /* For each image */
#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
        for (i = 0; i < n_images; i++)
        {
            int j;

            /* Get the local colormap if needed */
            if (g->SavedImages[i].ImageDesc.ColorMap)
            {

                /* TODO No support for local color map */
                /* fprintf(stderr, "Error: application does not support local colormap\n");
                 return NULL;*/

                colmap = g->SavedImages[i].ImageDesc.ColorMap;
            }
            /* Traverse the image and fill pixels */

            for (j = 0; j < width[i] * height[i]; j++)
            {
                int c;

                c = g->SavedImages[i].RasterBits[j];

                p[i][j] = (colmap->Colors[c].Red + (int)colmap->Colors[c].Green + (int)colmap->Colors[c].Blue) / 3;
            }
        }
    }
    /* Allocate image info */
    image = (animated_gif *)malloc(sizeof(animated_gif));
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

int output_modified_read_gif(char *filename, GifFileType *g)
{
    GifFileType *g2;
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

int store_pixels(char *filename, animated_gif *image)
{
    int n_colors = 256;
    pgrey **p;
    int i, j, k;
    GifColorType *colormap;

    /* Initialize the new set of colors */
    colormap = (GifColorType *)malloc(256 * sizeof(GifColorType));
    if (colormap == NULL)
    {
        fprintf(stderr,
                "Unable to allocate 256 colors\n");
        return 0;
    }

    // we only have 2 colors in the resulting gif
    for(i = 0; i < 256; i++){
        colormap[i].Red = i;
        colormap[i].Green = i;
        colormap[i].Blue = i;
    }

    image->g->SBackGroundColor = 128;

    /* Process extension blocks in main structure */

    for (j = 0; j < image->g->ExtensionBlockCount; j++)
    {
        int f;

        f = image->g->ExtensionBlocks[j].Function;
        if (f == GRAPHICS_EXT_FUNC_CODE)
        {
            image->g->ExtensionBlocks[j].Bytes[3] = 128;
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
                image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = 128;
            }
        }
    }

    p = image->p;

    /* Change the color map inside the animated gif */
    ColorMapObject *cmo;

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
            image->g->SavedImages[i].RasterBits[j] = p[i][j];
        }
    }

    /* Write the final image */
    if (!output_modified_read_gif(filename, image->g))
    {
        return 0;
    }

    return 1;
}
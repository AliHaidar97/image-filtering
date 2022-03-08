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
    pixel **p;
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
    p = (pixel **)malloc(n_images * sizeof(pixel *));
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
            p[i] = (pixel *)malloc(width[i] * height[i] * sizeof(pixel));
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

                p[i][j].r = colmap->Colors[c].Red;
                p[i][j].g = colmap->Colors[c].Green;
                p[i][j].b = colmap->Colors[c].Blue;
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
    int n_colors = 0;
    pixel **p;
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

    /* Everything is white by default */
    for (i = 0; i < 256; i++)
    {
        colormap[i].Red = 255;
        colormap[i].Green = 255;
        colormap[i].Blue = 255;
    }

    /* Change the background color and store it */
    int moy;
    moy = (image->g->SColorMap->Colors[image->g->SBackGroundColor].Red +
           image->g->SColorMap->Colors[image->g->SBackGroundColor].Green +
           image->g->SColorMap->Colors[image->g->SBackGroundColor].Blue) /
          3;
    if (moy < 0)
        moy = 0;
    if (moy > 255)
        moy = 255;

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
                    (image->g->SColorMap->Colors[tr_color].Red +
                     image->g->SColorMap->Colors[tr_color].Green +
                     image->g->SColorMap->Colors[tr_color].Blue) /
                    3;
                if (moy < 0)
                    moy = 0;
                if (moy > 255)
                    moy = 255;

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
                        moy == colormap[k].Red &&
                        moy == colormap[k].Green &&
                        moy == colormap[k].Blue)
                    {
                        found = k;
                    }
                }
                if (found == -1)
                {
                    if (n_colors >= 256)
                    {
                        fprintf(stderr,
                                "Error: Found too many colors inside the image\n");
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
                        (image->g->SColorMap->Colors[tr_color].Red +
                         image->g->SColorMap->Colors[tr_color].Green +
                         image->g->SColorMap->Colors[tr_color].Blue) /
                        3;
                    if (moy < 0)
                        moy = 0;
                    if (moy > 255)
                        moy = 255;

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
                            moy == colormap[k].Red &&
                            moy == colormap[k].Green &&
                            moy == colormap[k].Blue)
                        {
                            found = k;
                        }
                    }
                    if (found == -1)
                    {
                        if (n_colors >= 256)
                        {
                            fprintf(stderr,
                                    "Error: Found too many colors inside the image\n");
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
                            "Error: Found too many colors inside the image\n");
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
    if (!output_modified_read_gif(filename, image->g))
    {
        return 0;
    }

    return 1;
}
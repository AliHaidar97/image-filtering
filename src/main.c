/*
 * INF560
 *
 * Image Filtering Project
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <mpi.h>
#include <unistd.h>
#include <string.h>

#include "gif_lib.h"
#include "gif_load_store.h"
#include "cuda_sobelf.h"

// minimum number of nodes to switch from a even distribution approach to a server-client approach
#define MIN_NODES_SERVER_CLIENT 5

/*
void apply_gray_filter_img(pixel *p, int width, int height)
{
    int j;
#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
        for (j = 0; j < width * height; j++)
        {
            int moy;

            moy = (p[j].r + p[j].g + p[j].b) / 3;
            if (moy < 0)
                moy = 0;
            if (moy > 255)
                moy = 255;

            p[j].r = moy;
            p[j].g = moy;
            p[j].b = moy;
        }
    }
}

void apply_gray_filter(pixel **p, int n_images, int *widths, int *heights)
{
    int i, j;

    for (i = 0; i < n_images; i++)
    {
        apply_gray_filter_img(p[i], widths[i], heights[i]);
    }
}
*/

#define CONV(l, c, nb_c) \
    (l) * (nb_c) + (c)

void apply_blur_filter_img(pgrey *p, int width, int height, int size, int threshold)
{
    int end = 0;
    int n_iter = 0;

    int j, k;

    pgrey *new;

    n_iter = 0;

    /* Allocate array of new pixels */
    new = (pgrey *)malloc(width * height * sizeof(pgrey));

    /* Perform at least one blur iteration */
    do
    {
        end = 1;
        n_iter++;

#pragma omp parallel
        {
#pragma omp for schedule(dynamic) collapse(2)
            for (j = 0; j < height; j++)
            {
                for (k = 0; k < width; k++)
                {
                    new[CONV(j, k, width)] = p[CONV(j, k, width)];
                }
            }
        }

        /* Apply blur on the image (10%) */

#pragma omp parallel
        {
#pragma omp for schedule(dynamic) collapse(2)
            for (j = size; j < height - size; j++)
            {
                for (k = size; k < width - size; k++)
                {
                    int stencil_j, stencil_k;
                    int t = 0;

                    for (stencil_j = -size; stencil_j <= size; stencil_j++)
                    {
                        for (stencil_k = -size; stencil_k <= size; stencil_k++)
                        {
                            t += p[CONV(j + stencil_j, k + stencil_k, width)];
                        }
                    }

                    new[CONV(j, k, width)] = t / ((2 * size + 1) * (2 * size + 1));
                }
            }
        }

#pragma omp parallel
        {
#pragma omp for schedule(dynamic) collapse(2)
            for (j = 1; j < height - 1; j++)
            {
                for (k = 1; k < width - 1; k++)
                {

                    float diff;

                    diff = (new[CONV(j, k, width)] - p[CONV(j, k, width)]);

                    if (diff > threshold || -diff > threshold)
                    {
                        end = 0;
                    }

                    p[CONV(j, k, width)] = new[CONV(j, k, width)];
                }
            }
        }
    } while (threshold > 0 && !end);

#if SOBELF_DEBUG
    printf("BLUR: number of iterations for image %d\n", n_iter);
#endif

    free(new);
}

void apply_blur_filter(pgrey **p, int n_images, int *widths, int *heights, int size, int threshold)
{
    int i;
    /* Process all images */
    for (i = 0; i < n_images; i++)
    {
        int top_size = heights[i] / 10;
        // apply blur on the top part
        apply_blur_filter_img(p[i], widths[i], top_size, size, threshold);
        int bottom_size = heights[i] - (int)(heights[i] * 0.9);
        int bottom_start = heights[i] * 0.9;
        // apply blur on the bottom part
        apply_blur_filter_img(p[i] + widths[i] * bottom_start, widths[i], bottom_size, size, threshold);
    }
}

void apply_sobel_filter_img(pgrey *p, int width, int height)
{

    int j, k;
    pgrey *sobel;

    sobel = (pgrey *)malloc(width * height * sizeof(pgrey));

#pragma omp parallel
    {
#pragma omp for schedule(dynamic) collapse(2)
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

                pixel_blue_no = p[CONV(j - 1, k - 1, width)];
                pixel_blue_n = p[CONV(j - 1, k, width)];
                pixel_blue_ne = p[CONV(j - 1, k + 1, width)];
                pixel_blue_so = p[CONV(j + 1, k - 1, width)];
                pixel_blue_s = p[CONV(j + 1, k, width)];
                pixel_blue_se = p[CONV(j + 1, k + 1, width)];
                pixel_blue_o = p[CONV(j, k - 1, width)];
                pixel_blue = p[CONV(j, k, width)];
                pixel_blue_e = p[CONV(j, k + 1, width)];

                deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2 * pixel_blue_o + 2 * pixel_blue_e - pixel_blue_so + pixel_blue_se;

                deltaY_blue = pixel_blue_se + 2 * pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2 * pixel_blue_n - pixel_blue_no;

                val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue) / 4;

                if (val_blue > 50)
                {
                    sobel[CONV(j, k, width)] = 255;
                }
                else
                {
                    sobel[CONV(j, k, width)] = 0;
                }
            }
        }
    }

#pragma omp parallel
    {
#pragma omp for schedule(dynamic) collapse(2)
        for (j = 1; j < height - 1; j++)
        {
            for (k = 1; k < width - 1; k++)
            {
                p[CONV(j, k, width)] = sobel[CONV(j, k, width)];
            }
        }
    }
    free(sobel);
}

void apply_sobel_filter(pgrey **p, int n_images, int *widths, int *heights)
{
    int i;

    for (i = 0; i < n_images; i++)
    {

        apply_sobel_filter_img(p[i], widths[i], heights[i]);
    }
}

int rank, nb_proc;

const int tag_ready = 10000;
const int tag_send_size = 10001;
const int tag_send_dim = 10002;
const int tag_send_top = 10003;
const int tag_send_bottom = 10004;
const int tag_send_middle_part = 10005;
const int tag_sobelf_complete = 10006;

typedef struct Request
{
    int image_nb;
    int height_start;
    int height;
} Request;

const int blur_size = 5;
const int blur_threshold = 20;

// code executed by node 0, distribute tasks to clients
void server(int argc, char **argv)
{
    char *input_filename;
    char *output_filename;
    animated_gif *image;
    struct timeval t1, t2;
    double duration;
    int i;

    input_filename = argv[1];
    output_filename = argv[2];

    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Load file and store the pixels in array */
    image = load_pixels(input_filename);
    if (image == NULL)
    {
        return;
    }

    /* IMPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("GIF loaded from file %s with %d image(s) in %lf s\n",
           input_filename, image->n_images, duration);

    /* FILTER Timer start */
    gettimeofday(&t1, NULL);

    Request *requests = (Request *)malloc(nb_proc * sizeof(Request));
    for (int i = 0; i < nb_proc; i++)
    {
        // this proc is not doing any request right now
        requests[i].image_nb = -1;
    }

    // first send all top and bottom parts
    
    for (int i = 0; i < image->n_images; i++)
    {
        for (int k = 0; k < 2; k++)
        {
            MPI_Status stat;
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
            Request req = requests[stat.MPI_SOURCE];
            if (req.image_nb != -1)
            {
                // receive img
                pgrey *loc = image->p[req.image_nb] + req.height_start * image->width[req.image_nb];
                MPI_Recv(loc, req.height * image->width[req.image_nb] * sizeof(pgrey),
                         MPI_BYTE, stat.MPI_SOURCE, stat.MPI_TAG, MPI_COMM_WORLD, &stat);
            }
            else
            {
                // TODO: can we cancel it instead?
                // empty req
                MPI_Recv(NULL, 0,
                         MPI_BYTE, stat.MPI_SOURCE, stat.MPI_TAG, MPI_COMM_WORLD, &stat);
            }

            pgrey *start_loc;
            int height;
            int recv_height;
            int recv_height_start;
            int tag;
            if (k == 0)
            {
                start_loc = image->p[i];
                recv_height = image->height[i] / 10;
                height = recv_height + 1;
                recv_height_start = 0;
                tag = tag_send_bottom;
            }
            else
            {
                start_loc = image->p[i] + image->width[i] * (int)(image->height[i] * 0.9);
                start_loc -= image->width[i];
                recv_height = image->height[i] - (image->height[i] * 0.9);
                height = recv_height + 1;
                recv_height_start = (int)(image->height[i] * 0.9);
                tag = tag_send_top;
            }
            // first send dims
            int img_dims[] = {image->width[i], height};
            MPI_Send(img_dims, 2, MPI_INT, stat.MPI_SOURCE, tag_send_dim, MPI_COMM_WORLD);
            // then pixels
            MPI_Send(start_loc, height * image->width[i] * sizeof(pgrey),
                     MPI_BYTE, stat.MPI_SOURCE, tag, MPI_COMM_WORLD);

            requests[stat.MPI_SOURCE].image_nb = i;
            requests[stat.MPI_SOURCE].height_start = recv_height_start;
            requests[stat.MPI_SOURCE].height = recv_height;
        }
    }

    // then send all the remaining parts of the picture
    // get a send size so that each proc does not get too many requests
    int best_height = (image->height[0] * image->n_images) / (3 * nb_proc);
    if (best_height < 10)
    {
        // not too small
        best_height = 10;
    }

    for (int i = 0; i < image->n_images; i++)
    {
        int curr_height = image->height[i] / 10;
        int end_eight = image->height[i] * 0.9;
        while (curr_height < end_eight)
        {
            MPI_Status stat;
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
            Request req = requests[stat.MPI_SOURCE];
            if (req.image_nb != -1)
            {
                // receive img
                pgrey *loc = image->p[req.image_nb] + req.height_start * image->width[req.image_nb];
                MPI_Recv(loc, req.height * image->width[req.image_nb] * sizeof(pgrey),
                         MPI_BYTE, stat.MPI_SOURCE, stat.MPI_TAG, MPI_COMM_WORLD, &stat);
            }
            else
            {
                // TODO: can we cancel it instead?
                // empty req
                MPI_Recv(NULL, 0,
                         MPI_BYTE, stat.MPI_SOURCE, stat.MPI_TAG, MPI_COMM_WORLD, &stat);
            }

            pgrey *start_loc = image->p[i] + image->width[i] * (curr_height - 1);
            int height = best_height;
            if (height > end_eight - curr_height)
            {
                height = end_eight - curr_height;
            }

            int send_height = height + 2;

            // first send dims
            int img_dims[] = {image->width[i], send_height};
            MPI_Send(img_dims, 2, MPI_INT, stat.MPI_SOURCE, tag_send_dim, MPI_COMM_WORLD);
            // then pixels
            MPI_Send(start_loc, send_height * image->width[i] * sizeof(pgrey),
                     MPI_BYTE, stat.MPI_SOURCE, tag_send_middle_part, MPI_COMM_WORLD);

            requests[stat.MPI_SOURCE].image_nb = i;
            requests[stat.MPI_SOURCE].height_start = curr_height;
            requests[stat.MPI_SOURCE].height = height;

            curr_height += height;
        }
    }

    // make sure we received all images
    for (int r = 1; r < nb_proc; r++)
    {
        if (requests[r].image_nb != -1)
        {
            MPI_Status stat;
            MPI_Probe(r, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
            Request req = requests[r];
            if (req.image_nb != -1)
            {
                // receive img
                pgrey *loc = image->p[req.image_nb] + req.height_start * image->width[req.image_nb];
                MPI_Recv(loc, req.height * image->width[req.image_nb] * sizeof(pgrey),
                         MPI_BYTE, stat.MPI_SOURCE, stat.MPI_TAG, MPI_COMM_WORLD, &stat);
            }
            else
            {
                // empty req
                MPI_Recv(NULL, 0,
                         MPI_BYTE, stat.MPI_SOURCE, stat.MPI_TAG, MPI_COMM_WORLD, &stat);
            }
        }

        // tell proc we are done
        MPI_Send(NULL, 0, MPI_INT, r, tag_sobelf_complete, MPI_COMM_WORLD);
    }

    free(requests);

    /* FILTER Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("SOBEL done in %lf s\n", duration);

    /* EXPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Store file from array of pixels to GIF file */
    if (!store_pixels(output_filename, image))
    {
        return;
    }

    /* EXPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("Export done in %lf s in file %s\n", duration, output_filename);
}

// code executed by all mpi nodes != 0, apply sobel filters on parts of the image received
void client()
{
    int i;

    // send msg telling the server it is ready
    MPI_Send(NULL, 0, MPI_INT, 0, tag_ready, MPI_COMM_WORLD);

    // while no sobelf_complete tag is sent
    while (1)
    {
        MPI_Status stat;
        int dims[2];
        MPI_Recv(dims, 2, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);

        if (stat.MPI_TAG == tag_sobelf_complete)
        {
            return;
        }
        // stat.tag should be mpi_send_dims

        int width = dims[0];
        int height = dims[1];
        pgrey *p = (pgrey *)malloc(width * height * sizeof(pgrey));
        MPI_Recv(p, width * height * sizeof(pgrey), MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);

        /* Convert the pixels into grayscale */
        // apply_gray_filter_img(p, width, height);

        /* Apply blur filter with convergence value */
        int position = 0;
        if (stat.MPI_TAG == tag_send_bottom)
        {
            //apply_blur_filter_img(p, width, height - 1, blur_size, blur_threshold);
            position = 1;
        }
        else if (stat.MPI_TAG == tag_send_top)
        {
            //apply_blur_filter_img(p + width, width, height - 1, blur_size, blur_threshold);
            position = 2;
        }

        apply_filter_cuda(p, width, height, position, blur_size, blur_threshold);

        /* Apply sobel filter on pixels */
        apply_sobel_filter_img(p, width, height);

        pgrey *send_p = p;
        int send_height = height;

        if (stat.MPI_TAG != tag_send_bottom)
        {
            send_p += width;
            send_height--;
        }
        if (stat.MPI_TAG != tag_send_top)
        {
            send_height--;
        }

        MPI_Send(send_p, width * send_height * sizeof(pgrey), MPI_BYTE, 0, tag_ready, MPI_COMM_WORLD);

        free(p);
    }
}

void even_distribution(int argc, char **argv)
{
    char *input_filename;
    char *output_filename;
    animated_gif *image;
    struct timeval t1, t2;
    double duration;
    int i;

    pgrey **p;
    int n_images;
    int *widths, *heights;
    int *distribution;
    if (rank == 0)
    {

        input_filename = argv[1];
        output_filename = argv[2];

        /* IMPORT Timer start */
        gettimeofday(&t1, NULL);

        /* Load file and store the pixels in array */
        image = load_pixels(input_filename);
        if (image == NULL)
        {
            return;
        }

        /* IMPORT Timer stop */
        gettimeofday(&t2, NULL);

        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

        printf("GIF loaded from file %s with %d image(s) in %lf s\n",
               input_filename, image->n_images, duration);

        /* FILTER Timer start */
        gettimeofday(&t1, NULL);

        // on envoie en premier le nombre d'images par processeur
        distribution = (int *)malloc(nb_proc * sizeof(int));
        int *offsets = (int *)malloc(nb_proc * sizeof(int));
        offsets[0] = 0;
        int reste = image->n_images % nb_proc;
        for (i = 0; i < nb_proc; i++)
        {
            distribution[i] = (image->n_images / nb_proc) + (i < reste);
            if (i > 0)
            {
                offsets[i] = offsets[i - 1] + distribution[i - 1];
            }
        }
        MPI_Scatter(distribution, 1, MPI_INT, &n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // send width and height
        widths = (int *)malloc(n_images * sizeof(int));
        heights = (int *)malloc(n_images * sizeof(int));

        MPI_Scatterv(image->width, distribution, offsets, MPI_INT, widths, n_images, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(image->height, distribution, offsets, MPI_INT, heights, n_images, MPI_INT, 0, MPI_COMM_WORLD);

        // send images one by one
        int image_idx = distribution[0];
        for (i = 1; i < nb_proc; i++)
        {
            int k;
            for (k = 0; k < distribution[i]; k++)
            {
                // TODO: do this asynchronously
                MPI_Send(image->p[image_idx], (image->width[image_idx] * image->height[image_idx]) * sizeof(pgrey),
                         MPI_BYTE, i, 0, MPI_COMM_WORLD);
                image_idx++;
            }
        }

        p = image->p;

        free(offsets);
    }
    else
    {
        MPI_Scatter(NULL, 0, MPI_INT, &n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);

        widths = (int *)malloc(n_images * sizeof(int));
        heights = (int *)malloc(n_images * sizeof(int));

        MPI_Scatterv(NULL, NULL, NULL, MPI_INT, widths, n_images, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT, heights, n_images, MPI_INT, 0, MPI_COMM_WORLD);

        p = (pgrey **)malloc(n_images * sizeof(pgrey *));
        for (i = 0; i < n_images; i++)
        {
            p[i] = (pgrey *)malloc(widths[i] * heights[i] * sizeof(pgrey));
            MPI_Status stat;
            MPI_Recv(p[i], widths[i] * heights[i] * sizeof(pgrey), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &stat);
        }
    }

    /* Convert the pixels into grayscale */
    // apply_gray_filter(p, n_images, widths, heights);

    /* Apply blur filter with convergence value */
    apply_blur_filter(p, n_images, widths, heights, blur_size, blur_threshold);

    /* Apply sobel filter on pixels */
    apply_sobel_filter(p, n_images, widths, heights);

    // on recupere toutes les images sur le proc 0
    if (rank == 0)
    {
        int image_idx = distribution[0];
        for (i = 1; i < nb_proc; i++)
        {
            int k;
            for (k = 0; k < distribution[i]; k++)
            {
                MPI_Status stat;
                MPI_Recv(image->p[image_idx], (image->width[image_idx] * image->height[image_idx]) * sizeof(pgrey),
                         MPI_BYTE, i, 0, MPI_COMM_WORLD, &stat);
                image_idx++;
            }
        }
    }
    else
    {
        for (i = 0; i < n_images; i++)
        {
            MPI_Send(p[i], widths[i] * heights[i] * sizeof(pgrey),
                     MPI_BYTE, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0)
    {
        /* FILTER Timer stop */
        gettimeofday(&t2, NULL);

        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

        printf("SOBEL done in %lf s\n", duration);

        /* EXPORT Timer start */
        gettimeofday(&t1, NULL);

        /* Store file from array of pixels to GIF file */
        if (!store_pixels(output_filename, image))
        {
            return;
        }

        /* EXPORT Timer stop */
        gettimeofday(&t2, NULL);

        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

        printf("Export done in %lf s in file %s\n", duration, output_filename);
    }
}

/*
 * Main entry point
 */
int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Check command-line arguments */
    if (rank == 0 && argc < 3)
    {
        fprintf(stderr, "Usage: %s input.gif output.gif \n", argv[0]);
        return 1;
    }

    if (nb_proc < MIN_NODES_SERVER_CLIENT)
    {
        even_distribution(argc, argv);
    }
    else
    {
        if (rank == 0)
        {
            server(argc, argv);
        }
        else
        {
            client();
        }
    }

    MPI_Finalize();

    return 0;
}
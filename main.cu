#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>

#define MAX_CHAR_PER_LINE 128

float getRandWithRange(float min, float max);
cudaError_t kMeansCUDA(float *objects, float *clusters, int *membership, int N, int K, int dim_number, float threshold);
void kMeansCPU(float *objects, float *clusters, int K, int N, int dim_number, float threshold);
float *file_read(char *filename, int *N, int *dim_number, float **minInDim, float **maxInDim);
int file_write(char *filename, int K, int N, int dim_number, float *clusters, int *membership);
__host__ __device__ float distance_without_sqrt(float *x, float *y, int dim_number);

__global__ void kernel(float *objects, float *clusters, int *membership, int *delta, int N, int K, int dim_number)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        int index = 0;
        int dmin = distance_without_sqrt(objects + dim_number * i, clusters, dim_number);
        for (int j = 0; j < K; j++)
        {
            int distance = distance_without_sqrt(objects + dim_number * i, clusters + dim_number * j, dim_number);
            if (distance < dmin)
            {
                dmin = distance;
                index = j;
            }
        }

        if (membership[i] != index)
        {
            delta[i]++;
            membership[i] = index;
        }
    }
}

__global__ void kernel2(int *new_cluster_size, float *clusters, float *new_clusters, int *membership, int N, int K, int dim_number)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < K)
    {
        // int new_cluster_size = 0;
        // for (int j = 0; j < N; j++)
        // {
        //     if (membership[j] == i)
        //     {
        //         new_cluster_size++;
        //         for (int k = 0; k < dim_number; k++)
        //         {
        //             new_clusters[i * dim_number + k] += objects[j * dim_number + k];
        //         }
        //     }
        // }

        for (int j = 0; j < dim_number; j++)
        {
            clusters[i * dim_number + j] = new_clusters[i * dim_number + j] / new_cluster_size[i];
            new_clusters[i * dim_number + j] = 0;
        }
        new_cluster_size[i] = 0;
    }
}

__global__ void kernel3(float *new_clusters, float *objects, int *membership, int *new_cluster_size, int N, int dim_number)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        for (int j = 0; j < dim_number; j++)
        {
            atomicAdd(&new_clusters[membership[i]], objects[i * dim_number + j]);
            atomicAdd(&new_cluster_size[i], 1);
        }
    }
}

// __global__ void reduce(int N, int *key, float *out)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < N)
//     {
//         int li = ti % warpSize;

//         // code which computes a value per thread to be reduced,
//         // for threads where key[i] == -1, val = 0.f
//         float val = 123.45f;

//         // reset out to zero ready for atomic addition later
//         if (li == 0)
//             out[key[i]] = 0.f;
//         __syncthreads();

//         // warp reduction
//         int offset = 0;
//         for (offset = warpSize / 2; offset > 0; offset /= 2)
//             val += __shfl_down(val, offset);

//         // atomically update out with each warp's result
//         if (li == 0)
//             atomicAdd(&out[key[i]], val);
//     }
// }

int main(int argc, char **argv)
{
    if (argc != 3)
        return -1;
    char *filename = argv[1];
    int K = atoi(argv[2]);

    int N, dim_number;
    float *minInDim, *maxInDim;
    float threshold = 0.001;
    float *objects = file_read(filename, &N, &dim_number, &minInDim, &maxInDim);

    srand(time(0));
    float *clusters = (float *)malloc(K * dim_number * sizeof(float));
    assert(clusters != NULL);
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < dim_number; j++)
        {
            clusters[i * dim_number + j] = getRandWithRange(minInDim[j], maxInDim[j]);
        }
    }
    int *membership = (int *)malloc(N * sizeof(int));
    assert(membership != NULL);

    // printf("I'm starting counting on CPU\n");
    // kMeansCPU(objects, clusters, K, N, dim_number, threshold);

    printf("I'm starting counting on GPU\n");
    cudaError_t cudaStatus = kMeansCUDA(objects, clusters, membership, N, K, dim_number, threshold);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kMeansCuda failed!\n");
        return 1;
    }

    file_write(filename, K, N, dim_number, clusters, membership);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!\n");
        return 1;
    }

    free(objects);
    free(clusters);
    free(minInDim);
    free(maxInDim);

    return 0;
}

cudaError_t kMeansCUDA(float *objects, float *clusters, int *membership, int N, int K, int dim_number, float threshold)
{
    float *dev_objects, *dev_clusters, *dev_newclusters;
    int *dev_membership, *dev_delta, *dev_newclustersize, *dev_fake, *dev_membershipToReduce;
    int numberOfBlocks = (N - 1) / 1024 + 1, loop = 0, numberOfBlocks2 = (K - 1) / 1024 + 1, i;
    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time, deltaSum = 1000000;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
        goto Error;
    }

    // dev_objects
    cudaStatus = cudaMalloc((void **)&dev_objects, N * dim_number * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_objects, objects, N * dim_number * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Error;
    }

    // dev_clusters
    cudaStatus = cudaMalloc((void **)&dev_clusters, K * dim_number * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_clusters, clusters, K * dim_number * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemset failed!\n");
        goto Error;
    }

    // dev_newclusters
    cudaStatus = cudaMalloc((void **)&dev_newclusters, K * dim_number * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }

    cudaStatus = cudaMemset(dev_newclusters, 0, K * dim_number * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemset failed!\n");
        goto Error;
    }

    // dev_membership
    cudaStatus = cudaMalloc((void **)&dev_membership, N * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }

    cudaStatus = cudaMemset(dev_membership, 0, N * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemset failed!\n");
        goto Error;
    }

    // dev_membershipToReduce
    cudaStatus = cudaMalloc((void **)&dev_membershipToReduce, N * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }

    cudaStatus = cudaMemset(dev_membershipToReduce, 0, N * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemset failed!\n");
        goto Error;
    }

    // dev_newclustersize
    cudaStatus = cudaMalloc((void **)&dev_newclustersize, K * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }

    cudaStatus = cudaMemset(dev_newclustersize, 0, K * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemset failed!\n");
        goto Error;
    }

    // dev_delta
    cudaStatus = cudaMalloc((void **)&dev_delta, N * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }

    // dev_fake
    cudaStatus = cudaMalloc((void **)&dev_fake, N * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }

    cudaStatus = cudaMemset(dev_fake, 1, N * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemset failed!\n");
        goto Error;
    }

    cudaEventRecord(start, 0);
    do
    {
        cudaStatus = cudaMemset(dev_delta, 0, N * sizeof(int));
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemset failed!\n");
            goto Error;
        }

        kernel<<<numberOfBlocks, 1024>>>(dev_objects, dev_clusters, dev_membership, dev_delta, N, K, dim_number);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        // kernel3<<<numberOfBlocks, 1024>>>(dev_newclusters, dev_objects, dev_membership, dev_newclustersize, N, dim_number);
        for (i = 0; i < dim_number; i++)
        {
            thrust::reduce_by_key(thrust::device, dev_clusters + N * i, dev_clusters + N * (i + 1), dev_membership, NULL, dev_newclusters);
        }
        thrust::reduce_by_key(thrust::device, dev_fake, dev_fake + N, dev_membership, NULL, dev_newclustersize);

        kernel2<<<numberOfBlocks2, 1024>>>(dev_newclustersize, dev_clusters, dev_newclusters, dev_membership, N, K, dim_number);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "kernel2 launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        deltaSum = thrust::reduce(thrust::device, dev_delta, dev_delta + N, (float)0);
        printf("loop nr %d\n", loop);
    } while (deltaSum / N > threshold && loop++ < 500);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Counting on GPU lasts: %fs\n", time / 1000);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(clusters, dev_clusters, K * dim_number * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Error;
    }

    cudaStatus = cudaMemcpy(membership, dev_membership, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Error;
    }
Error:
    cudaFree(dev_objects);
    cudaFree(dev_clusters);
    cudaFree(dev_newclusters);
    cudaFree(dev_membership);

    return cudaStatus;
}

float getRandWithRange(float min, float max)
{
    float value = rand() / (float)RAND_MAX;
    return min + value * (max - min);
}

// N - number of objects, K - number of clusters
void kMeansCPU(float *objects, float *clusters, int K, int N, int dim_number, float threshold)
{
    int delta, index, dmin, loop = 0;

    int *membership = (int *)malloc(N * sizeof(int));
    assert(membership != NULL);
    memset(membership, 0, N);

    float *new_clusters = (float *)malloc(K * dim_number * sizeof(float));
    assert(new_clusters != NULL);
    memset(new_clusters, 0, K * dim_number);

    int *new_cluster_size = (int *)malloc(K * sizeof(int));
    assert(new_cluster_size != NULL);
    memset(new_cluster_size, 0, K);

    do
    {
        delta = 0;
        for (int i = 0; i < N; i++)
        {
            index = 0;
            dmin = distance_without_sqrt(objects + dim_number * i, clusters, dim_number);
            for (int j = 0; j < K; j++)
            {
                int distance = distance_without_sqrt(objects + dim_number * i, clusters + dim_number * j, dim_number);
                if (distance < dmin)
                {
                    dmin = distance;
                    index = j;
                }
            }

            if (membership[i] != index)
            {
                delta++;
                membership[i] = index;
            }

            new_cluster_size[index]++;
            for (int j = 0; j < dim_number; j++)
            {
                new_clusters[index * dim_number + j] += objects[i * N + j];
            }
        }

        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < dim_number; j++)
            {
                clusters[i * dim_number + j] = new_clusters[i * dim_number + j] / new_cluster_size[i];
                new_clusters[i * dim_number + j] = 0;
            }
            new_cluster_size[i] = 0;
        }
    } while (delta / N > threshold && loop++ < 500);

    free(membership);
    free(new_clusters);
    free(new_cluster_size);
}

float *file_read(char *filename, int *N, int *dim_number, float **minInDim, float **maxInDim)
{
    float *objects;
    int i, j, len;

    FILE *infile;
    char *line, *ret;
    const char *delim = " \t\n\r";
    int lineLen;

    if ((infile = fopen(filename, "r")) == NULL)
    {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        return NULL;
    }

    lineLen = MAX_CHAR_PER_LINE;
    line = (char *)malloc(lineLen);
    assert(line != NULL);

    (*N) = 0;
    while (fgets(line, lineLen, infile) != NULL)
    {
        /* check each line to find the max line length */
        while (strlen(line) == lineLen - 1)
        {
            /* this line read is not complete */
            len = (int)strlen(line);
            fseek(infile, -len, SEEK_CUR);

            /* increase lineLen */
            lineLen += MAX_CHAR_PER_LINE;
            line = (char *)realloc(line, lineLen);
            assert(line != NULL);

            ret = fgets(line, lineLen, infile);
            assert(ret != NULL);
        }

        if (strtok(line, delim) != 0)
            (*N)++;
    }
    rewind(infile);

    (*dim_number) = 0;
    while (fgets(line, lineLen, infile) != NULL)
    {
        if (strtok(line, delim) != 0)
        {
            /* ignore the id (first coordiinate): numCoords = 1; */
            while (strtok(NULL, delim) != NULL)
                (*dim_number)++;
            break; /* this makes read from 1st object */
        }
    }
    rewind(infile);

    printf("%d, %d\n", *N, *dim_number);
    objects = (float *)malloc((*N) * (*dim_number) * sizeof(float));
    assert(objects != NULL);

    (*minInDim) = (float *)malloc(*dim_number * sizeof(float));
    assert(minInDim != NULL);
    (*maxInDim) = (float *)malloc(*dim_number * sizeof(float));
    assert(maxInDim != NULL);
    for (int i = 0; i < *dim_number; i++)
    {
        (*minInDim)[i] = FLT_MAX;
        (*maxInDim)[i] = FLT_MIN;
    }

    i = 0;
    while (fgets(line, lineLen, infile) != NULL)
    {
        if (strtok(line, delim) == NULL)
            continue;
        for (j = 0; j < (*dim_number); j++)
        {
            float value = atof(strtok(NULL, delim));
            objects[i * (*dim_number) + j] = value;
            if ((*minInDim)[j] > value)
            {
                (*minInDim)[j] = value;
            }
            if ((*maxInDim)[j] < value)
            {
                (*maxInDim)[j] = value;
            }
        }
        i++;
    }

    fclose(infile);
    free(line);

    return objects;
}

int file_write(char *filename, int K, int N, int dim_number, float *clusters, int *membership)
{
    FILE *file;
    int i, j;
    char outFileName[1024];

    /* output: the coordinates of the cluster centres ----------------------*/
    sprintf(outFileName, "%s.cluster_centres", filename);
    printf("Writing coordinates of K=%d cluster centers to file \"%s\"\n",
           K, outFileName);
    file = fopen(outFileName, "w");
    for (i = 0; i < K; i++)
    {
        fprintf(file, "%d ", i);
        for (j = 0; j < dim_number; j++)
            fprintf(file, "%f ", clusters[i * K + j]);
        fprintf(file, "\n");
    }
    fclose(file);

    /* output: the closest cluster centre to each of the data points --------*/
    sprintf(outFileName, "%s.membership", filename);
    printf("Writing membership of N=%d data objects to file \"%s\"\n",
           N, outFileName);
    file = fopen(outFileName, "w");
    for (i = 0; i < N; i++)
        fprintf(file, "%d %d\n", i, membership[i]);
    fclose(file);

    return 1;
}

__host__ __device__ float distance_without_sqrt(float *x, float *y, int dim_number)
{
    float result = 0.0;

    for (int i = 0; i < dim_number; i++)
    {
        result += (x[i] - y[i]) * (x[i] - y[i]);
    }

    return result;
}
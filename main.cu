#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>

#define MAX_CHAR_PER_LINE 128

float getRandWithRange(float min, float max);
cudaError_t kMeansCUDA(float *objects, float *clusters, int *membership, int N, int K, int dim_number, float threshold);
void kMeansCPU(float *objects, float *clusters, int *membership, int K, int N, int dim_number, float threshold);
float *file_read(char *filename, int *N, int *dim_number, float **minInDim, float **maxInDim);
int file_write(char *filename, const char *version, int K, int N, int dim_number, float *clusters, int *membership);
__host__ __device__ float distance_without_sqrt(float *object, float *cluster, int dim_number, int object_i, int cluster_i, int N, int K);

__global__ void kernel(float *objects, float *clusters, int *membership, int *delta, int N, int K, int dim_number)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        int index = 0;
        float dmin = distance_without_sqrt(objects, clusters, dim_number, i, 0, N, K);
        for (int j = 0; j < K; j++)
        {
            float distance = distance_without_sqrt(objects, clusters, dim_number, i, j, N, K);
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

__global__ void kernel2(int *new_cluster_size, float *clusters, float *new_clusters, int *keys, int K, int dim_number)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < K)
    {
        int index = keys[i];
        if (index == -1)
            return;
        keys[i] = -1;
        if (new_cluster_size[i] != 0)
        {
            for (int j = 0; j < dim_number; j++)
            {
                clusters[index + j * K] = new_clusters[i + j * K] / new_cluster_size[i];
                new_clusters[i + j * K] = 0;
            }
            new_cluster_size[i] = 0;
        }
    }
}

__global__ void kernel3(float *clusters, float *new_clusters, float *objects, int *membership, int N, int K, int dim_number)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < K)
    {
        int new_cluster_size = 0;
        for (int j = 0; j < N; j++)
        {
            if (membership[j] == i)
            {
                new_cluster_size++;
                for (int k = 0; k < dim_number; k++)
                {
                    new_clusters[i + k * K] += objects[j + k * N];
                }
            }
        }

        for (int j = 0; j < dim_number; j++)
        {
            clusters[i + j * K] = new_clusters[i + j * K] / new_cluster_size;
            new_clusters[i + j * K] = 0;
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 4)
    {
        printf("Podaj: nazwa_pliku liczba_klastrów (CPU)\n");
        return -1;
    }
    char *filename = argv[1];
    bool CPU = false;
    int K = atoi(argv[2]);
    if (K == 0)
    {
        printf("Podaj: nazwa_pliku liczba_klastrów (CPU)\n");
        return -1;
    }

    if (argc == 4 && strcmp(argv[3], "CPU") == 0)
    {
        CPU = true;
    }

    int N, dim_number;
    float *minInDim, *maxInDim;
    float threshold = 0.001;
    float *objects = file_read(filename, &N, &dim_number, &minInDim, &maxInDim);

    srand(10);
    float *clusters = (float *)malloc(K * dim_number * sizeof(float));
    assert(clusters != NULL);
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < dim_number; j++)
        {
            clusters[i + j * K] = getRandWithRange(minInDim[j], maxInDim[j]);
        }
    }

    int *membership = (int *)malloc(N * sizeof(int));
    assert(membership != NULL);
    memset(membership, -1, N * sizeof(int));

    if (CPU)
    {
        float *clustersCPU = (float *)malloc(K * dim_number * sizeof(float));
        assert(clustersCPU != NULL);
        memcpy(clustersCPU, clusters, K * dim_number * sizeof(float));

        printf("I'm starting counting on CPU\n");
        kMeansCPU(objects, clustersCPU, membership, K, N, dim_number, threshold);
        file_write(filename, "CPU", K, N, dim_number, clustersCPU, membership);
        free(clustersCPU);
    }

    printf("I'm starting counting on GPU\n");
    cudaError_t cudaStatus = kMeansCUDA(objects, clusters, membership, N, K, dim_number, threshold);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kMeansCuda failed!\n");
        return 1;
    }
    file_write(filename, "GPU", K, N, dim_number, clusters, membership);

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
    free(membership);
    free(minInDim);
    free(maxInDim);

    return 0;
}

cudaError_t kMeansCUDA(float *objects, float *clusters, int *membership, int N, int K, int dim_number, float threshold)
{
    float *dev_objects, *dev_clusters, *dev_newclusters, *dev_objectsToSort;
    int *dev_membership, *dev_delta, *dev_newclustersize, *dev_membershipToSort, *dev_keys;
    int numberOfBlocks = (N - 1) / 1024 + 1, loop = 0, numberOfBlocks2 = (K - 1) / 1024 + 1, i;
    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time, deltaSum;

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

    // dev_objectsToSort
    cudaStatus = cudaMalloc((void **)&dev_objectsToSort, N * dim_number * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\n");
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

    cudaStatus = cudaMemset(dev_membership, -1, N * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemset failed!\n");
        goto Error;
    }

    // dev_membershipToSort
    cudaStatus = cudaMalloc((void **)&dev_membershipToSort, N * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\n");
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

    // dev_keys
    cudaStatus = cudaMalloc((void **)&dev_keys, dim_number * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }

    cudaStatus = cudaMemset(dev_keys, -1, dim_number * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemset failed!\n");
        goto Error;
    }

    cudaEventRecord(start, 0);
    do
    {
        if (loop != 0)
        {
            cudaStatus = cudaMemcpy(dev_objectsToSort, dev_objects, N * dim_number * sizeof(float), cudaMemcpyDeviceToDevice);
            if (cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "cudaMemcpy failed!\n");
                goto Error;
            }

            for (i = 0; i < dim_number; i++)
            {
                cudaStatus = cudaMemcpy(dev_membershipToSort, dev_membership, N * sizeof(int), cudaMemcpyDeviceToDevice);
                if (cudaStatus != cudaSuccess)
                {
                    fprintf(stderr, "cudaMemcpy failed!\n");
                    goto Error;
                }

                thrust::sort_by_key(thrust::device, dev_membershipToSort, dev_membershipToSort + N, dev_objectsToSort + N * i);
                thrust::reduce_by_key(thrust::device, dev_membershipToSort, dev_membershipToSort + N, dev_objectsToSort + N * i, dev_keys, dev_newclusters + K * i);
            }

            cudaStatus = cudaMemcpy(dev_membershipToSort, dev_membership, N * sizeof(int), cudaMemcpyDeviceToDevice);
            if (cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "cudaMemcpy failed!\n");
                goto Error;
            }

            thrust::sort(thrust::device, dev_membershipToSort, dev_membershipToSort + N);
            thrust::reduce_by_key(thrust::device, dev_membershipToSort, dev_membershipToSort + N, thrust::make_constant_iterator(1), dev_keys, dev_newclustersize);

            kernel2<<<numberOfBlocks2, 1024>>>(dev_newclustersize, dev_clusters, dev_newclusters, dev_keys, K, dim_number);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "kernel2 launch failed: %s\n", cudaGetErrorString(cudaStatus));
                goto Error;
            }
        }

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

        deltaSum = thrust::reduce(thrust::device, dev_delta, dev_delta + N, (float)0);

        printf("loop nr %d, deltaSum: %d\n", loop, (int)deltaSum);
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
    cudaFree(dev_objectsToSort);
    cudaFree(dev_clusters);
    cudaFree(dev_newclusters);
    cudaFree(dev_membership);
    cudaFree(dev_membershipToSort);
    cudaFree(dev_keys);
    cudaFree(dev_delta);
    cudaFree(dev_newclustersize);

    return cudaStatus;
}

float getRandWithRange(float min, float max)
{
    float value = rand() / (float)RAND_MAX;
    return min + value * (max - min);
}

// N - number of objects, K - number of clusters
void kMeansCPU(float *objects, float *clusters, int *membership, int K, int N, int dim_number, float threshold)
{
    int delta, index, loop = 0;

    float *new_clusters = (float *)malloc(K * dim_number * sizeof(float));
    assert(new_clusters != NULL);
    memset(new_clusters, 0, K * dim_number * sizeof(float));

    int *new_cluster_size = (int *)malloc(K * sizeof(int));
    assert(new_cluster_size != NULL);
    memset(new_cluster_size, 0, K * sizeof(int));

    do
    {
        delta = 0;
        for (int i = 0; i < N; i++)
        {
            index = 0;
            float dmin = distance_without_sqrt(objects, clusters, dim_number, i, 0, N, K);
            for (int j = 0; j < K; j++)
            {
                float distance = distance_without_sqrt(objects, clusters, dim_number, i, j, N, K);
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
                new_clusters[index + j * K] += objects[i + j * N];
            }
        }

        for (int i = 0; i < K; i++)
        {
            if (new_cluster_size[i] != 0)
            {
                for (int j = 0; j < dim_number; j++)
                {
                    clusters[i + j * K] = new_clusters[i + j * K] / new_cluster_size[i];
                    new_clusters[i + j * K] = 0;
                }
                new_cluster_size[i] = 0;
            }
        }

        printf("loop nr %d, deltaSum: %d\n", loop, delta);
    } while ((float)delta / N > threshold && loop++ < 500);

    free(new_clusters);
    free(new_cluster_size);
}

float distance_without_sqrt(float *object, float *cluster, int dim_number, int object_i, int cluster_i, int N, int K)
{
    float result = 0.0;

    for (int i = 0; i < dim_number; i++)
    {
        result += (object[object_i + i * N] - cluster[cluster_i + i * K]) * (object[object_i + i * N] - cluster[cluster_i + i * K]);
    }

    return result;
}

float *file_read(char *filename, int *N, int *dim_number, float **minInDim, float **maxInDim)
{
    float *objects;
    int i, j, len;

    FILE *file;
    char *line, *ret;
    const char *delim = " \t\n\r";
    int lineLen;

    if ((file = fopen(filename, "r")) == NULL)
    {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        return NULL;
    }

    lineLen = MAX_CHAR_PER_LINE;
    line = (char *)malloc(lineLen);
    assert(line != NULL);

    (*N) = 0;
    while (fgets(line, lineLen, file) != NULL)
    {
        /* check each line to find the max line length */
        while (strlen(line) == lineLen - 1)
        {
            /* this line read is not complete */
            len = (int)strlen(line);
            fseek(file, -len, SEEK_CUR);

            /* increase lineLen */
            lineLen += MAX_CHAR_PER_LINE;
            line = (char *)realloc(line, lineLen);
            assert(line != NULL);

            ret = fgets(line, lineLen, file);
            assert(ret != NULL);
        }

        if (strtok(line, delim) != 0)
            (*N)++;
    }
    rewind(file);

    (*dim_number) = 0;
    while (fgets(line, lineLen, file) != NULL)
    {
        if (strtok(line, delim) != 0)
        {
            /* ignore the id (first coordiinate): numCoords = 1; */
            while (strtok(NULL, delim) != NULL)
                (*dim_number)++;
            break; /* this makes read from 1st object */
        }
    }
    rewind(file);

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
    while (fgets(line, lineLen, file) != NULL)
    {
        if (strtok(line, delim) == NULL)
            continue;
        for (j = 0; j < (*dim_number); j++)
        {
            float value = atof(strtok(NULL, delim));
            objects[i + j * (*N)] = value;
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

    fclose(file);
    free(line);

    return objects;
}

int file_write(char *filename, const char *version, int K, int N, int dim_number, float *clusters, int *membership)
{
    FILE *file;
    int i, j;
    char outFileName[1024];

    /* output: the coordinates of the cluster centres ----------------------*/
    sprintf(outFileName, "%s.%s.cluster_centres.txt", filename, version);
    printf("Writing coordinates of K=%d cluster centers to file \"%s\"\n",
           K, outFileName);
    file = fopen(outFileName, "w");
    for (i = 0; i < K; i++)
    {
        fprintf(file, "%d ", i);
        for (j = 0; j < dim_number; j++)
            fprintf(file, "%f ", clusters[i + j * K]);
        fprintf(file, "\n");
    }
    fclose(file);

    /* output: the closest cluster centre to each of the data points --------*/
    sprintf(outFileName, "%s.%s.membership.txt", filename, version);
    printf("Writing membership of N=%d data objects to file \"%s\"\n",
           N, outFileName);
    file = fopen(outFileName, "w");
    for (i = 0; i < N; i++)
        fprintf(file, "%d %d\n", i, membership[i]);
    fclose(file);

    return 1;
}
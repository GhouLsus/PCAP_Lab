#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

void err_handle(int err_code)
{
    if (err_code != MPI_SUCCESS)
    {
        char err_string[1000];
        int length;
        MPI_Error_string(err_code, err_string, &length);
        fprintf(stderr, "MPI Error %s\n", err_string);
        MPI_Abort(MPI_COMM_WORLD, err_code);
        return;
    }
}

int main(int argc, char **argv)
{
    int rank, size, *arr, *store;
    int err_code = MPI_Init(&argc, &argv);
    err_handle(err_code);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
    {
        arr = (int *)malloc(16 * sizeof(int));
        printf("enter the elements\n");
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                scanf("%d", &arr[(4 * i) + j]);
            }
        }
    }
    store = (int *)malloc(4 * sizeof(int));
    err_code = MPI_Scatter(arr, 4, MPI_INT, store, 4, MPI_INT, 0, MPI_COMM_WORLD);
    err_handle(err_code);
    for (int i = 0; i < 4; ++i)
    {
        err_code = MPI_Scan(&store[i], &store[i], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        err_handle(err_code);
    }
    err_code = MPI_Gather(store, 4, MPI_INT, arr, 4, MPI_INT, 0, MPI_COMM_WORLD);
    err_handle(err_code);
    if (rank == 0)
    {
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                printf("%d ", arr[(4 * i) + j]);
            }
            printf("\n");
        }
    }
    MPI_Finalize();
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define MATRIX_SIZE 3
#define TAG 0
int count_occurrences(int *row, int target) {
    int count = 0;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        if (row[i] == target) {
            count++;
        }
    }
    return count;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[MATRIX_SIZE][MATRIX_SIZE];
    int target, local_count = 0, global_count = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 3) {
        if (rank == 0) {
            printf("This program requires exactly 3 processes.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    if (rank == 0) {

        printf("Enter the elements of a 3x3 matrix:\n");
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }

        printf("Enter the element to be searched: ");
        scanf("%d", &target);
    }

    MPI_Bcast(&target, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(matrix, MATRIX_SIZE * MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    int row[MATRIX_SIZE];
    MPI_Scatter(matrix, MATRIX_SIZE, MPI_INT, row, MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    local_count = count_occurrences(row, target);

    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("The element %d appears %d times in the matrix.\n", target, global_count);
    }

    MPI_Finalize();
    return 0;
}

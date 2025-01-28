#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

long long factorial(int n) {
    long long result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

int main(int argc, char** argv) {
    int rank, size, N;
    MPI_Init(&argc, &argv);                
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
    MPI_Comm_size(MPI_COMM_WORLD, &size);  

    if (rank == 0) { 
        printf("Enter the number of values (N): ");
        scanf("%d", &N);

        if (N != size) {
            printf("Error: Number of values (N) must match the number of processes!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int* values = (int*)malloc(N * sizeof(int));
        printf("Enter %d values:\n", N);
        for (int i = 0; i < N; i++) {
            scanf("%d", &values[i]);
        }

        for (int i = 0; i < N; i++) {
            printf("Root (Rank 0): Sending value %d to Rank %d\n", values[i], i);
            MPI_Send(&values[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        free(values);
    }

    int value;
    MPI_Recv(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Process (Rank %d): Received value %d from Root\n", rank, value);

    long long fact = factorial(value);
    printf("Process (Rank %d): Factorial of %d = %lld\n", rank, value, fact);

    MPI_Send(&fact, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);

    if (rank == 0) { 
        long long sum = 0;
        for (int i = 0; i < size; i++) {
            long long received_fact;
            MPI_Recv(&received_fact, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Root (Rank 0): Received factorial %lld from Rank %d\n", received_fact, i);
            sum += received_fact;
        }
        printf("Root (Rank 0): Sum of factorials = %lld\n", sum);
    }

    MPI_Finalize();
    return 0;
}

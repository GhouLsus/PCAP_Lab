#include "mpi.h"
#include <stdio.h>

int power(int x, int exp){
    int ans = 1;
    for (int i = 0; i < exp; i++){
        ans *= x;
    }
    return ans;
}

int main(int argc, char** argv){
    int size;
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Rank is %d, Size is %d, Power of rank raised to size is %d\n", rank, size, power(rank, size));

    MPI_Finalize();
    return 0;
}

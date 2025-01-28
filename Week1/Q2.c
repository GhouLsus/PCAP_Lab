# include "mpi.h"
# include<stdio.h>

void main(int argc , char** argv){
    int size;
    int rank;

    MPI_Init(&argc , &argv);
    MPI_Comm_rank(MPI_COMM_WORLD , &rank);
    MPI_Comm_size(MPI_COMM_WORLD ,&size);

    if(rank % 2 == 0){
        printf("rank %d hello\n", rank);
    }
    else{
printf("rank %d world\n" , rank);
    }

    MPI_Finalize();
}
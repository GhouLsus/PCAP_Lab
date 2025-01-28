#include<mpi.h> 
#include <stdio.h>
#include<string.h>
int main (int argv , char*argc[]){
    int rank , size ; 
    MPI_Init(&argv , &argc); 
    MPI_Comm_rank( MPI_COMM_WORLD , &rank);
    MPI_Comm_size( MPI_COMM_WORLD , &size);
    char st[100];
    int len ; 
    printf("Rank : %d \n" ,rank);
    MPI_Status status ; 
    if (rank==0){   
        scanf("%s" , st);
        len = strlen(st);
        MPI_Send( &len , 1 , MPI_INT , 1 ,  0 , MPI_COMM_WORLD );
        // +1 for the /0
        MPI_Send( st , len+1 , MPI_CHAR , 1 , 1 , MPI_COMM_WORLD );
        MPI_Recv( st , len+1 , MPI_CHAR ,  1 ,  2 , MPI_COMM_WORLD , &status);
        printf("rank : %d  , toggled string : %s",rank,st);
    }
    else { 
        MPI_Recv( &len , 1 , MPI_INT ,  0 ,  0 , MPI_COMM_WORLD , &status);
        MPI_Recv( st , len+1 , MPI_CHAR ,  0 ,  1 , MPI_COMM_WORLD , &status);
        
        for (int i = 0 ; i < len ; i++){
            // printf("%s \n" , st);
            st[i]^=32 ;
        }
        MPI_Send( st , len+1 , MPI_CHAR , 0 , 2 , MPI_COMM_WORLD );
    
    }
    MPI_Finalize();
    return 0 ;

}
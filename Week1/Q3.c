#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

                                                                                                                                     
void perform_addition(int rank, double a, double b) {
    printf("Process %d: Addition of %.2f and %.2f is %.2f\n", rank, a, b, a + b);
}


void perform_subtraction(int rank, double a, double b) {
    printf("Process %d: Subtraction of %.2f and %.2f is %.2f\n", rank, a, b, a - b);
}


void perform_multiplication(int rank, double a, double b) {
    printf("Process %d: Multiplication of %.2f and %.2f is %.2f\n", rank, a, b, a * b);
}


void perform_division(int rank, double a, double b) {
    if (b != 0) {
        printf("Process %d: Division of %.2f by %.2f is %.2f\n", rank, a, b, a / b);
    } else {
        printf("Process %d: Division by zero is not allowed.\n", rank);
    }
}

int main(int argc, char** argv) {
    int rank, size;
    double a = 10.0, b = 5.0;

    
    MPI_Init(&argc, &argv);

    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 4) {
        if (rank == 0) {
            printf("Please run the program with at least 4 processes.\n");
        }
        MPI_Finalize();
        return 0;
    }

    
    switch (rank) {
        case 0:
            perform_addition(rank, a, b);
            break;
        case 1:
            perform_subtraction(rank, a, b);
            break;
        case 2:
            perform_multiplication(rank, a, b);
            break;
        case 3:
            perform_division(rank, a, b);
            break;
        default:
            printf("Process %d: No operation assigned.\n", rank);
            break;
    }


    MPI_Finalize();
    return 0;
}


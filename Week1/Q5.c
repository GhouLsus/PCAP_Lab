#include <mpi.h>
#include <stdio.h>


long long factorial(int n) {
    long long result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}


long long fibonacci(int n) {
    if (n <= 1) return n;
    long long a = 0, b = 1, c;
    for (int i = 2; i <= n; i++) {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}

int main(int argc, char** argv) {
    int rank, size;

    
    MPI_Init(&argc, &argv);

    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank % 2 == 0) {
        
        long long fact = factorial(rank);
        printf("Process %d (even): Factorial of %d is %lld\n", rank, rank, fact);
    } else {
        
        long long fib = fibonacci(rank);
        printf("Process %d (odd): Fibonacci number for %d is %lld\n", rank, rank, fib);
    }

    
    MPI_Finalize();
    return 0;
}


# Assignment #1: Profiling using gprof and working with mpic++

This is my first assignment for the parallel algorithms course. In the following, I'll try to explain the project and the setup as comprehensively as possible. The goal of this assignment is
to experience working with tools that enable us to analyze parallelism and multicore
processing across different applications. The codes available here are derived from [here](https://github.com/gordon-cs/cps343-hoe/tree/master/00-intro-to-hpc) with little modification. In this particular assignment, we are first going to
analyze matrix multiplication and then analyze an approximation on Pi value using
4/(1+x^2) integral. 

Before we start, let’s have a look at my system specifications for comparison purposes and
just to make sure everything is set up. First, my CPU specification is:






```
Architecture:            x86_64
CPU op-mode(s):          32-bit, 64-bit
Address sizes:           39 bits physical, 48 bits virtual
CPU(s):                  8
On-line CPU(s) list:     0-7
Thread(s) per core:      2
Core(s) per socket:      4
Socket(s):               1
NUMA node(s):            1
CPU family:              6
Model name:              Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz
CPU MHz:                 2000.000
CPU max MHz:             4000.0000
CPU min MHz:             4000.0000
L1d cache:               128 KiB
L1i cache:               128 KiB
L2 cache:                1 MiB
L3 cache:                8 MiB
```
And my compiler versions are:
```
gcc (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
mpirun (Open MPI) 4.0.3
GNU gprof (GNU Binutils for Ubuntu) 2.34
```
## Part One: Matrix Matrix Multiplication

After we make sure that everything is set, we move to the `matrix_matrix_mulyiply` folder and
invoke the `makefile.` This can be done by putting the command `$ make` in the terminal. And
you get:

```g++ -Wall -O3 -funroll-loops    matmat_ijk.cc -o    matmat_ijk```

If every dependency is already satisfied and the files are compiled, you should get something like this as output:

```make: Nothing to be done for 'all'.```

It’s time to compile the `matmat_ijk.cc` file which contains matrix matrix multiplication.
Concretely, multiplying two 2D matrices with size n*n is of O(n^3), where there is a three for
loop overall (AB = C):

```c++
for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
        for (int k = 0; k < n; k++)
            c[i][j] = c[i][j] + a[i][k]*b[k][j];
```

We can work around this algorithm and change the loop order. That is, we can change the
loop order from ijk all the way to kji which results in 6 different combinations. The objective is
to find out the difference in execution time to see which loop order is faster. The file
`matmat_ijk.cc` contains only one combination so we append the other 5 combinations to this
file and use the `$ make` command again. In order to compile this file we type the following
command in the terminal:

```$ ./matmat_ijk```

This should output:

```
Matrix-Matrix multiply (1D Array w/macro): Matrices are 500x500
ijk:   0.143989 sec,     1736.24 mflops,   checksum =    31274290.412853
ikj:   0.035322 sec,     7077.73 mflops,   checksum =    31274290.412853
jik:   0.133942 sec,     1866.48 mflops,   checksum =    31274290.412853
jki:   0.143781 sec,     1738.76 mflops,   checksum =    31274290.412853
kij:   0.036188 sec,     6908.28 mflops,   checksum =    31274290.412853
kji:   0.148148 sec,     1687.50 mflops,   checksum =    31274290.412853
```

Where the matrices’ size is 500, execution time is listed in front of every combination in
seconds, number of floating point operations is given in millions and finally checksum is also computed in order to make sure that rearranging loop order does not change the final
multiplication result.

The important question that comes to mind at first is: why is the execution time and the
number of floating point operations per second different among different loop orders? Let’s
run this file using gprof, a performance analysis tool to see the differences more accurately. We
will closely look at 3 loop order for matrix matrix multiplication in the file `matmat_cxx.cc`.
For this, just put in the command below:

```$ gprof matmat_cxx```

The output should briefly look like this:

```
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  ms/call  ms/call  name    
 36.41      0.58     0.58        1   582.57   582.57  matmat_ijk(double*, double*, double*, int)
 35.78      1.16     0.57        1   572.53   572.53  matmat_jki(double*, double*, double*, int)
 28.25      1.61     0.45        1   451.99   451.99  matmat_ikj(double*, double*, double*, int)
  0.00      1.61     0.00        6     0.00     0.00  wtime()
  0.00      1.61     0.00        3     0.00     0.00  verify(double*, double*, int)
```

Where `% time` shows the percentage of the total running time of the program used by this
function, `cumulative seconds` shows a running sum of the number of seconds accounted for
by this function and those listed above it. The `self seconds` shows the number of seconds
accounted for by this function alone. This is the major sort for this listing. The gprof tool lists
these evaluations based on the portion of time that each function gets to spend on the whole
program run.

As we can observe, the loop order ikj has the least execution time which means it has the best
performance among all other combinations. The reason for this lies within the access time to
on-chip memory (i.e., cache), that is, in order to multiply each row of matrix A to each column
of matrix B, the cpu must access matrix cells of these matrices. Therefore, the closer the faster
the time to fetch these cells, the faster the execution time. This can be further discussed with
locality of reference. Now, one last detail - in C/C++, arrays are stored in row-major order,
which means that all of the values in a single row of a matrix are stored next to each other.
Thus in memory the array looks like the first row, then the second row, then the third row, etc.
Given this, let's look at your code. The first version looks like this:

```c++
  for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
          for (int k = 0; k < n; k++)
              c[i][j] = c[i][j] + a[i][k]*b[k][j];
```

Now, let's look at that innermost line of code. On each iteration, the value of k is increasing. This means that when running the innermost loop, each iteration of the loop is likely to have a cache miss when loading the value of b[k][j]. The reason for this is that because the matrix is
stored in row-major order, each time you increment k, you're skipping over an entire row of
the matrix and jumping much further into memory, possibly far past the values you've cached.
However, you don't have a miss when looking up `c[i][j]` (since i and j are the same), nor will
you probably miss `a[i][k]`, because the values are in row-major order and if the value of
`a[i][k]` is cached from the previous iteration, the value of `a[i][k]` read on this iteration is
from an adjacent memory location. Consequently, on each iteration of the innermost loop, you
are likely to have one cache miss. But consider this second version:

```c++
  for (int i = 0; i < n; i++)
      for (int k = 0; k < n; k++)
          for (int j = 0; j < n; j++)
              c[i][j] = c[i][j] + a[i][k]*b[k][j];
```

Now, since you're increasing j on each iteration, let's think about how many cache misses you'll likely have on the innermost statement. Because the values are in row-major order, the value of `c[i][j]` is likely to be in-cache, because the value of `c[i][j]` from the previous iteration is likely cached as well and ready to be read. Similarly, `b[k][j]` is probably cached, and since i and k aren't changing, chances are `a[i][k]` is cached as well. This means that on each iteration of the inner loop, you're likely to have no cache misses.

Overall, this means that the second version of the code is unlikely to have cache misses on each iteration of the loop, while the first version almost certainly will. Consequently, the second loop is likely to be faster than the first, as you've seen.

Now, Let’s change N (matrix dimension size) to 200, 500 and 1000 to analyze execution time.
For N = 200:

```
Matrix-Matrix multiply (1D Array w/macro): Matrices are 200x200
ijk:   0.012844 sec,     1245.75 mflops,   checksum =     2000904.610452
ikj:   0.003119 sec,     5129.39 mflops,   checksum =     2000904.610452
jik:   0.010856 sec,     1473.87 mflops,   checksum =     2000904.610452
jki:   0.007637 sec,     2095.06 mflops,   checksum =     2000904.610452
kij:   0.002580 sec,     6202.42 mflops,   checksum =     2000904.610452
kji:   0.007487 sec,     2137.11 mflops,   checksum =     2000904.610452
```

For N = 500:

```
Matrix-Matrix multiply (1D Array w/macro): Matrices are 500x500
ijk:   0.142656 sec,     1752.47 mflops,   checksum =    31218278.051027
ikj:   0.033761 sec,     7404.94 mflops,   checksum =    31218278.051027
jik:   0.135670 sec,     1842.71 mflops,   checksum =    31218278.051027
jki:   0.146261 sec,     1709.27 mflops,   checksum =    31218278.051027
kij:   0.034900 sec,     7163.31 mflops,   checksum =    31218278.051027
kji:   0.147333 sec,     1696.83 mflops,   checksum =    31218278.051027
```

And finally, for N = 1000:

```
Matrix-Matrix multiply (1D Array w/macro): Matrices are 1000x1000
ijk:   1.905969 sec,     1049.34 mflops,   checksum =   249762932.138095
ikj:   0.410760 sec,     4869.03 mflops,   checksum =   249762932.138095
jik:   1.491354 sec,     1341.06 mflops,   checksum =   249762932.138095
jki:   7.708052 sec,      259.47 mflops,   checksum =   249762932.138095
kij:   0.437609 sec,     4570.29 mflops,   checksum =   249762932.138095
kji:   7.108104 sec,      281.37 mflops,   checksum =   249762932.138095
```

We can see as N gets larger the best execution time gets larger overall, but for larger Ns, the loop order ikj is the most optimal as opposed to loop order kij which is the most optimal for N = 200.

##  Part Two: Pi Approximation using Integral

 In the second part, we will evaluate the integral of 4/(1+x^2) on [0, 1] using the midpoint rule. We will first analyze the execution time of this program running serially on only one core, and then we will exploit multicore processing using mpic++. MPI is a directory of C++ programs which illustrate the use of the Message Passing Interface for parallel programming. MPI allows a user to write a program in a familiar language, such as C, C++, FORTRAN, or Python, and carry out a computation in parallel on an arbitrary number of cooperating computers.

 In order to make sure that mpic++ is up and running, we will run a simple hello world program across different numbers of cores to see the output. So, just to be clear the chunk of code we are dealing with is:

```c++
#include <iostream>
#include <mpi.h>


using namespace std;

int main(int argc, char **argv) {
  // Initialize MPI
  // This must always be called before any other MPI functions
  MPI_Init(&argc, &argv);

  // Get the number of processes in MPI_COMM_WORLD
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of this process in MPI_COMM_WORLD
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // Print out information about MPI_COMM_WORLD
  std::cout <<endl << "This is my first program using mpi:  "<< "World Size: " << world_size << "   Rank: " << my_rank << std::endl;

  // Finalize MPI
  // This must always be called after all other MPI functions
  MPI_Finalize();

  return 0;
}
```

In MPI, each copy or "process" is assigned a unique number, referred to as the rank of the process, and each process can obtain its rank when it runs. The copies of an MPI program, once they start running, must coordinate with one another somehow. This cooperation starts when each one calls an initialization function before it uses any other MPI features. The prototype for this function appears below:

  ```MPI_Init(&argc, &argv);```

As a rule of thumb, it is a good idea to call `MPI_Init` as the first statement of our program, and `MPI_Finalize` as its last statement. Let's now modify our "Hello, world!" program accordingly. `MPI_Comm_size` reports the number of processes running as part of this job by assigning it to the result parameter `world_size`. Similarly, `MPI_Comm_rank` reports the rank of the calling process to the result parameter `my_rank`.

Now, we compile and run this program using 1, 2 and 4  processors. Note that each running process produces output based on the values of its local variables. The stdout of all running processes is simply concatenated together. As we run the program using more processes, you may see that the output from the different processes does not appear in order or rank. Let’s compile our hello.cpp file using this command:

```$ mpic++ hello.cpp```

Then, we can run this c++ program using mipc++ while specifying the output file and the number of cores (i.e, -np 2) using this command:

```$ mpirun -np 2 ./a.out```

And finally the output should be like below for 1, 2 and 4 processors respectively:

```
This is my first program using mpi:  World Size: 1   Rank: 0

*********************************************

This is my first program using mpi:  World Size: 2   Rank: 1

This is my first program using mpi:  World Size: 2   Rank: 0

*********************************************

This is my first program using mpi:  World Size: 4   Rank: 2

This is my first program using mpi:  World Size: 4   Rank: 1

This is my first program using mpi:  World Size: 4   Rank: 3

This is my first program using mpi:  World Size: 4   Rank: 0
```

Now that we made sure everything is set and mipc++ is up and running, it is time to jump into our second assignment. First we compile and run the serial.cc file in order to be able to compare the single core serial run with the multicore parallel run. So, using the below command:

```$ ./pi_serial```

We get the following result:

```pi = 3.141592653589552 computed in 0.5288 seconds; rate = 5.295 GFLOPS```


An approximation for pi value computed in about 5 seconds at the rate 5.2 Giga floating point operations per second, of course, running in serial. Now, let’s use the mpic++ to exploit parallelism and see the difference. We are going to start by first running on two cores by putting in this commands respectively:

```$ mpic++ pi_mpi.cc```


```$ mpirun -np 2 ./a.out```


The output should look like this:

```pi = 3.141592653590022 computed in 1.431 seconds; rate = 1.957 GFLOPS```

Let’s try this with 4 cores (`-np 4`) to see the difference:

```pi = 3.141592653590041 computed in 0.7391 seconds; rate = 3.788 GFLOPS```


As we can see, exploiting more cores results in less time and more FLOPs. Concretely, when using only 2 cores to run in parallel the execution time is approximately two time as long as running with 4 cores, which makes sense.













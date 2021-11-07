
# Assignment #3: Matrix-Matrix Multiplication Speedup


## Background

We'll use OpenMP to create several different parallel applications, each one using a slightly different approach to parallelizing the work. 
Suppose that `A` is an `m×p` matrix and `B` is a `p×n` matrix. the product `C=AB` will be an `m×n` matrix. We've studied the six different `ijk` loop orderings to compute the product, and seen that performance is impacted by our choice. Assuming our matrices are stored in row-major form, the `ikj` order is one of the best. Here is some sample code for this. Note that we assume that the matrices are stored in 1-D arrays and that the macro `IDX(i,j,n)` has been defined:


```c
    for ( int i = 0; i < m * n; i++ )
    {
        c[i] = 0.0;
    }
    for ( int i = 0; i < m; i++ )
    {
        for ( int k = 0; k < p; k++ )
        {
            for ( int j = 0; j < n; j++ )
            {
                c[IDX(i,j,n)] += a[IDX(i,k,p)] * b[IDX(k,j,n)];
            }
        }
     }
```

 - Question: How could we partition this work into individual tasks? Perhaps the smallest obvious task is the computation of a single element of C. Thus, the tasks   could be indexed by i and j.

 - Question: Assuming this partition, what is the communcation flow between tasks? Since each `cij` is computed independently of the others, we don't really have any communication to worry about.

 - Question: How should we group the tasks? Notice that since we're using the `ikj` ordering, cij will be updated multiple times inside the outer i-loop. Notice however, that each iteration of the outer loop computes the entire ith row of C. Thus, a reasonable task grouping would be to create task groups responsible for computing a row (or multiple rows) of C.

 - Question: How can these task groups be assigned to threads (or processes)? Working this out is what this hands-on exercise is all about!


## Using OpenMP for our first parallel program

1- Change into the `Matrix Matrix Multiplication Speedup` directory.

2- Examine the code stored in `matmat_serial.cc`. You'll see it computes the product of two square matrices A and B. Actually it does this twice – you'll modify the source so the second product is computed in parallel. 

3- We're now going to jump right in and use OpenMP to introduce parallelism to this program. We'll actually do it two different ways!

4- Our first parallel program will use an `OpenMP` compiler directive to instruct the compiler to break the execution of a for-loop into multiple threads, as if each thread is responsible for the execution of the loop body for a single value of the loop variable.

 - Exercise: Copy the file `matmat_serial.cc` to `matmat_omp1.cc` and modify this new file using OpenMP compiler directives so that the second matrix product is computed using multiple threads. You will find it helpful to look at the `pi_omp.cc` program from our first  exercise. You may also want to take a couple minutes to check out the [LLNL OpenMP Tutorial](https://computing.llnl.gov/tutorials/openMP/).

You only need to make four changes to the code: 

  * add the flag `-fopenmp` to compilation command in the `$Smake` line 

  * add an include statment to include the `omp.h` file

  * add the compiler directive #pragma omp parallel for default(shared) statement before each of the two outer i loops in the `doParallelProduct()` function
  update comments in the code as necessary

  * Run your program with the following commands: 
```
OMP_NUM_THREADS=1 ./matmat_omp1
OMP_NUM_THREADS=2 ./matmat_omp1
OMP_NUM_THREADS=3 ./matmat_omp1
.
.
.
OMP_NUM_THREADS=8 ./matmat_omp1
```

 - (Suggestion: write a single-line bash script to do this) and observe the behavior. Also try running the program without setting `OMP_NUM_THREADS`. If this is not set, OpenMP will use the same number of threads as available CPU cores as reported by `nproc.`)

Wasn't that easy? This sort of thing is where `OpenMP` seems almost too good to be true. It works very well in this program because (1)~the ith row of C can be computed completely independently of every other row - there are no dependences to worry about. Since these operations account for virtually all the execution time for a matrix-matrix product, we can easily achieve nearly perfect speedup.

5- Now we'll try a different approach. This will be a little harder to implement, but does give us a taste for how we might implement a dynamic load balancing scheme.
As befere, we'll consider the computation of a row of C to be a task. Suppose our program will start N threads. Each thread will enter a critical section to determine what row it should compute, leave the critical section, then compute the row. It will continue to do this until it gets a row number that is not in the matrix.

 - Exercise: Copy `matmat_serial.cc` to `matmat_omp2.cc` and add the `-fopenmp` flag and `#include <omp.h>` statement as before.
Find the `doParallelProduct()` function and place a `#pragma omp parallel` for default(shared) directive immediate before first loop. This means that we will once again initialize C in parallel.
Replace the first two lines of the second for-loop 

```c
  for ( int i = 0; i < n; i++ )
  {		       
```

with the lines 

```
  int nextRow = 0;
  while ( true )
  {
      int i;
      i = nextRow++;
      if ( i >= n ) break;
```

Compile and run the program. You should find that the reported time for the ‘parallel’ version is about the same as the time for the serial version. The change we made does nothing to parallelize the product, but but does set us up for what we want to do next.

Edit `matmat_omp2.cc` again, and 
 - add the directive `#pragma omp parallel default(shared)` (note that this does not include the keyword for) immediately before the while-loop. This creates a thread pool with each thread executing the while-loop.
 - add the directive `#pragma omp critical` after the declaration of i and immediately before the statement `i = nextRow++;`. This causes that statement to be executed in a critical region – only one thread can do this at time.
 - Update and add comments as needed.

Compile and run the program again. Now you should see that the second product is computed in parallel.
Run your program with the following commands: 

```
OMP_NUM_THREADS=1 ./matmat_omp2
OMP_NUM_THREADS=2 ./matmat_omp2
OMP_NUM_THREADS=3 ./matmat_omp2
.
.
.
OMP_NUM_THREADS=8 ./matmat_omp2
```

## Results:

The highlighted numbers are the most significant speedup based on two different approach:

![alt text](https://raw.githubusercontent.com/amir-ghz/Multicore-and-GPU-Programming/main/Matrix%20Matrix%20Multiplication%20Speedup/table.png)

Also below is the chart to compare parallelism speedup comprehensively:

![alt text](https://raw.githubusercontent.com/amir-ghz/Multicore-and-GPU-Programming/main/Matrix%20Matrix%20Multiplication%20Speedup/speedup.png)
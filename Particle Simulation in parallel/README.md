
# Assignment #2: Particle Simulation in parallel

This is the second assignment for the parallel algorithms course. In this project, we simulate particle interaction, an application applicable to fields such as mechanics, biology, and astronomy. When particles reach a certain cutoff radius, they repel each other. A na¨ıve implementation of the serial algorithm computes the forces between all pairs of particles, which, for n particles, is of O(n^2) algorithm. A complete discription of the project can be find [here (Applications of Parallel Computing (CS 267) at Berkeley)](https://danieltakeshi.github.io/2016-05-27-review-of-applications-of-parallel-computing-cs-267-at-berkeley/)

## Part One: Running the algorithm in serial

In the second part, we will evaluate the execution time based on particle numbers. The below plot describes the number of particles and their execution time.

![alt text](https://raw.githubusercontent.com/amir-ghz/Multicore-and-GPU-Programming/main/Particle%20Simulation%20in%20parallel/results-part-1.png)


##  Part Two: Exploiting parallelism using openMP

In this section, we are going to use OpenMP to exploit parallelism. We parallelize our productive serial calculation utilizing OpenMP, which is based on a shared memory model. To begin with, we compute the 2D vector which relegates particles to bins. We are able to do this in parallel utilizing `#pragma omp parallel` for, making beyond any doubt to share the containers information structure, which will separate the particles among the threads. The work to compute the bin that a particular particle belongs to is the same as in the serial case. However, when updating our 2D vector, we put a `#pragma omp critical` to ensure no data races and write conflicts ensue. Since this is a parallel for-loop ranging over all n particles and does O(1) work within, the cost is O(log n). However, because of the critical section, each processor must perform this data write sequentially, causing the overall cost of this computation to be O(n). Thus, this computation is actually the bottleneck of our algorithm. (We attempted to use locks instead of a critical section in order to speed up the computation but were unable to debug the deadlocks or segmentation fault on the lock that we were getting. We are still mystified as to why this was occurring, and it would be nice to take a look at the lock usage in the code together to see if we have some simple error in how we are using OpenMP).

![alt text](https://raw.githubusercontent.com/amir-ghz/Multicore-and-GPU-Programming/main/Particle%20Simulation%20in%20parallel/results-part-2.png)


## Part Three: Exploiting parallelism using pthreads

Pthreads and OpenMP represent two totally different multiprocessing paradigms. Pthreads is a very low-level API for working with threads. Thus, you have extremely fine-grained control over thread management (create/join/etc), mutexes, and so on. It's fairly bare-bones.

On the other hand, OpenMP is much higher level, is more portable, and doesn't limit you to using C. It's also much more easily scaled than pthreads. One specific example of this is OpenMP's work-sharing constructs, which let you divide work across multiple threads with relative ease. (See also Wikipedia's pros and cons list.) That said, you've really provided no detail about the specific program you're implementing, or how you plan on using it, so it's fairly impossible to recommend one API over the other.

If you use OpenMP, it can be as simple as adding a single pragma, and you'll be 90% of the way to properly multithreaded code with linear speedup. To get the same performance boost with pthreads takes a lot more work.

But as usual, you get more flexibility with pthreads. Basically, it depends on what your application is. Do you have a trivially-parallelizable algorithm? Or do you just have lots of arbitrary tasks that you'd like to do simultaneously? How much do the tasks need to talk to each other? How much synchronization is required?

The key questions are- 1) "Does your codebase have threads, thread pools, and the control primitives (locks, events, etc.)" and 2) "Are you developing reusable libraries or ordinary apps?"

If your library has thread tools (almost always built on some flavor of PThread), USE THOSE. If you are a library developer, spend the time (if possible) to build them. It is worth it- you can put together much more fine-grained, advanced threading than OpenMP will give you. Conversely, if you are pressed for time or just developing apps or something off of 3rd party tools, use OpenMP. You can wrap it in a few macros and get the basic parallelism you need. In general, OpenMP is good enough for basic multi-threading. Once you start getting to the point that you're managing system resources directly on building highly async code, its ease-of-use advantage gets crowded out by performance and interface issues.


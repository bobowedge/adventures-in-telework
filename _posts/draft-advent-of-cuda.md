---
toc: true
layout: post
description: The one where I try to save Christmas from 6 years ago
categories: [markdown]
title: Advent of Cuda (2015)
---

# Advent of Cuda 

## Introduction

The vast majority of my previous posts have focused on my efforts with the [fastai MOOC](https://course.fast.ai/) that I had been working through. Since I had watched all the lessons for the course and worked through most of it, I moved onto the next thing I wanted to try as part of my telework journey: learning [CUDA](https://en.wikipedia.org/wiki/CUDA).[^1]

Rather than try to work through a textbook (e.g. [this one](https://www.amazon.com/CUDA-Example-Introduction-General-Purpose-Programming-ebook/dp/B003VYBOSE)) or a guided online tutorial, I decided to try something a little different. This past December, I was introduced to [Advent of Code](https://adventofcode.com/2020), which is a series of small programming puzzles released one a day from December 1 to December 25. For the 2020 version, I wrote all my solutions in Python on the day they were released to try to do them as fast as possible.[^2]  Prior year versions of Advent of Code are also available: I enjoyed the 2020 version so much that I started doing some 2015 problems in Python as well. 

### Setup 

And that's where we'll start this CUDA story. I previously did the first 17 days[^3] for Advent of Code 2015 in Python. Of those days, my Python solutions for days [4](https://adventofcode.com/2015/day/4), [9](https://adventofcode.com/2015/day/9), [10](https://adventofcode.com/2015/day/10), and [11](https://adventofcode.com/2015/day/11) were "slow": they took more than 5 seconds to run. I figured those problems would be worth trying to tackle with CUDA to speed up. 

To start, though, I decided to do days [1](https://adventofcode.com/2015/day/1) and [2](https://adventofcode.com/2015/day/2) in CUDA to get my feet wet. After that and the "slow" problems, I worked the remaining problems where I didn't have a Python safety net.

There's a [github repo](https://github.com/bobowedge/advent-of-code-2015) with all of the code that I wrote (C++ CUDA and Python) for solving the problems. Also included is the input data[^4] for the days that I didn't incorporate it directly into the code.

### Caveats

My point for doing this was to learn C++ CUDA and then try to explain it. That means I won't necessarily be taking the most efficient or direct approach to solving each problem:  I'm basically brand new to CUDA (so I won't know the "best" approach a priori) and I wanted to learn new parts of the C++ CUDA syntax/library (so I won't necessarily take the "best" approach even if I know it). 

I'm using my home computer that has an NVIDA GTX 1660, which seems to a moderate consumer GPU, though I'm sure some will disagree.[^5]  For this project, everything I did was restricted to a single device (my home GPU), so there's nothing multi-device here. Also, I didn't seem to run into any memory problems running the code that I wrote, but, of course, YMMV. I'm compiling using the `nvcc` compiler out of Visual Studio Code terminal in Windows.

Last caveat is that I'm a C++ programmer at heart, so I'm going to use C++ where I can and muddle my way through when I can't. Unfortunately, one of the first things that I learned was that none of the [C++ STL](https://en.wikipedia.org/wiki/Standard_Template_Library) is supported on CUDA device code, so no STL containers. :thumbsdown:

### Premise

For Advent of Code 2015, the premise is that Santa's weather machine can't produce snow because it lacks the ***stars*** required. Each programming part you solve earns a star and collect 50 (2 for each day) gives you enough to power the snow machine.  Let's see if I can save Christmas with CUDA.

## [Day 1: Not Quite Lisp](https://adventofcode.com/2015/day/1)

This problem boiled down to evaluating a string consisting of opening and closing parentheses, `+1` for each opening parenthesis in the string and `-1` for each closing parenthesis. The standard "hello world" example for CUDA is [vector addition](https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf)[^6], so I decided to adapt that to tackle this problem. The core device code (with multiple threads and multiple blocks) for summing two vectors , `a` and `b`, into the resultant vector `c` is

```c++
int index = threadIdx.x + blockIdx.x * blockDim.x;
while (index < N)
{
    c[index] = a[index] + b[index];
    index += blockDim.x * gridDim.x;
}
```
Each thread in each block takes a number of indexes in the arrays and does their sum. Which indexes each thread takes relies on these parameters:
- blockIdx.x :arrow_right: Index of the block (in the x-direction)
- threadIdx.x :arrow_right: Index of the thread in a block (in the x-direction)
- blockDim.x :arrow_right: Number of threads per block (in the x-direction)
- gridDim.x :arrow_right: Number of blocks (in the x-direction)

For my problem, the input is not integer arrays, but a string of `(` and `)`, that I'll denote as `instructions`.  The idea I had was that each thread could take some of the instructions and sum those up (`+1` for `(`, `-1` for `)`). The core device code becomes[^7]
```c++
int64_t sum = 0;
int64_t index = threadIdx.x + blockIdx.x * blockDim.x;
while (index < N)
{
    if (instructions[index] == '(')
    {
        ++sum;
    }
    else
    {
        --sum;
    }
    index += blockDim.x * gridDim.x;
}
```
Then, we need a way to combine the sums from all the threads and all the blocks. 

For combining all the threads, there's a common idiom for summing up the values from each thread. First, you create an array of shared block values (`__shared__`) holds the sum for each thread: 
```c++
__shared__ int64_t cache[THREADS];
const int64_t cacheIndex = threadIdx.x;
```

Then, there's a way to sum the values in the cache in parallel to give `cache[0]` the total sum of all the threads in the block. The term that I have seen for this is "reduction":
```c++
// Do a reduction to sum the elements created by each thread in a block 
__device__ void reduction(int64_t *cache, int64_t cacheIndex)
{
    int64_t index = blockDim.x >> 1;
    while (index > 0)
    {
        if (cacheIndex < index)
        {
            cache[cacheIndex] += cache[cacheIndex + index];
        }
        __syncthreads();
        index >>= 1;
    }
    return;
}
```

Putting it together (where `sum` is the sum from each thread):
```c++
__shared__ int64_t cache[THREADS];
const int64_t cacheIndex = threadIdx.x;
cache[cacheIndex] = sum;
// Sync every thread in this block
__syncthreads();

// Reduce cache to a single value
reduction(cache, cacheIndex);

if (cacheIndex == 0)
{
    result[blockIdx.x] = cache[0];
}
```
This gives the sum for each block. To combine the block sums, I returned that result array to the host and used `std::accumulate` to get the answer.

(I skipped writing Part 2 of Day 1 in CUDA: the problem was to find the first place the partial sum of `instructions` was negative (given the same `+1` and `-1` for `(` and `)`).  It seemed too serial for CUDA.)

## Day 2: I Was Told There Would Be No Math

BLAH BLAH BLAH

[^1]: I came close to learning some CUDA before: I took an High Performance Computing class two years ago that covered programming in CUDA for a day or two. Unfortunately, my daughter ran a fever at daycare that day, so I missed it and couldn't find the time to revisit it. Now, however, with the opportunity to learn it at my own pace, I had no excuses.

[^2]: At first, because I wanted to get on the leaderboard. However, it became quickly clear  that was never happening, but I kept up the facade of doing it as quickly as ***I*** could.

[^3]: Each day is a 2-part problem, where the second part is revealed after correctly answering the first part

[^4]: One of the cool things about Advent of Code is that each person gets a different batch of input data.

[^5]: Opinions are like my friends, they're all assholes. :fire:

[^6]:  In addition to being the example that NVIDIA uses, it's the first meaningful example in the book I mentioned above,and pretty much all of the video examples if you search for "Introduction to CUDA". 

[^7]: Ternary statements are the devil's code, in case you're wondering.
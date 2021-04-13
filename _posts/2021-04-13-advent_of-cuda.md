---
toc: true
layout: post
description: The one where I try to save Christmas from 6 years ago
categories: [markdown]
title: Advent of Cuda 2015
---

# Advent of Cuda 

## Introduction

The vast majority of my previous posts have focused on my efforts with the [fastai MOOC](https://course.fast.ai/) that I had been working through. Since I had watched all the lessons for the course and worked through most of it, I moved onto the next thing I wanted to try as part of my telework journey: learning [CUDA](https://en.wikipedia.org/wiki/CUDA).<sup id="a1">[1](#f1)</sup>

Rather than try to work through a textbook (e.g. [this one](https://www.amazon.com/CUDA-Example-Introduction-General-Purpose-Programming-ebook/dp/B003VYBOSE)) or a guided online tutorial, I decided to try something a little different. This past December, I was introduced to [Advent of Code](https://adventofcode.com/2020), which is a series of small programming puzzles released one a day from December 1 to December 25. For the 2020 version, I wrote all my solutions in Python on the day they were released to try to do them as fast as possible.<sup id="a2">[2](#f2)</sup>  Prior year versions of Advent of Code are also available: I enjoyed the 2020 version so much that I started doing some 2015 problems in Python as well. 

### Setup 

And that's where we'll start this CUDA story. I previously did the first 17 days<sup id="a3">[3](#f3)</sup> for Advent of Code 2015 in Python. Of those days, my Python solutions for days [4](https://adventofcode.com/2015/day/4), [9](https://adventofcode.com/2015/day/9), [10](https://adventofcode.com/2015/day/10), and [11](https://adventofcode.com/2015/day/11) were "slow": they took more than 5 seconds to run. I figured those problems would be worth trying to tackle with CUDA to speed up. 

To start, though, I decided to do days [1](https://adventofcode.com/2015/day/1) and [2](https://adventofcode.com/2015/day/2) in CUDA to get my feet wet. After that and the "slow" problems, I worked the remaining problems where I didn't have a Python safety net.

There's a [github repo](https://github.com/bobowedge/advent-of-code-2015) with all of the code that I wrote (C++ CUDA and Python) for solving the problems. Also included is the input data<sup id="a4">[4](#f4)</sup> for the days that I didn't incorporate it directly into the code.

I would also be remiss if I didn't mention that I used the [NVIDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) pretty extensively. The guide is fairly comprehensive and useful for looking stuff up.

### Caveats

My point for doing this was to learn C++ CUDA and then try to explain it. That means I won't necessarily be taking the most efficient or direct approach to solving each problem:  I'm basically brand new to CUDA, so I won't know the "best" approach a priori. Also, I wanted to learn new parts of the C++ CUDA syntax/library, so I won't necessarily take the "best" approach even if I know it. 

My goal with the writing and explanation of each problem below are to write it at the level that someone who has seen the standard [vector addition example](https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf)<sup id="a5">[5](#f5)</sup> can follow. That's the level I was at when I started Day 1 in CUDA.

I'm using my home computer that has an NVIDA GTX 1660, which seems to a moderate consumer GPU, though I'm sure some will disagree.<sup id="a6">[6](#f6)</sup>  For this project, everything I did was restricted to a single device (my home GPU), so there's nothing multi-device here. Also, I didn't seem to run into any memory problems running the code that I wrote, but, of course, YMMV. I'm compiling using the `nvcc` compiler out of Visual Studio Code terminal in Windows.

I'm a C++ programmer at heart, so I'm going to use C++ where I can and muddle my way through when I can't. Unfortunately, one of the first things that I learned was that none of the [C++ STL](https://en.wikipedia.org/wiki/Standard_Template_Library) is supported on CUDA device code, so no STL containers. :thumbsdown:

### Premise

For Advent of Code 2015, the premise is that Santa's weather machine can't produce snow because it lacks the stars required. Each programming part you solve earns a star and collect 50 (2 for each day) gives you enough to power the snow machine.  Let's see if I can save Christmas with CUDA.

## [Day 1: Not Quite Lisp](https://adventofcode.com/2015/day/1)

### Part 1

This problem boiled down to evaluating a string consisting of opening and closing parentheses, `+1` for each opening parenthesis in the string and `-1` for each closing parenthesis. I decided to adapt the vector addition example to tackle this problem, since it was the only thing I had so far. For that problem, the core device code (with multiple threads and multiple blocks) for summing two vectors , `a` and `b`, into the resultant vector `c` is

```c++
int index = threadIdx.x + blockIdx.x * blockDim.x;
while (index < N)
{
    c[index] = a[index] + b[index];
    index += blockDim.x * gridDim.x;
}
```
Each thread in each block takes a number of indexes in the arrays and does their sum. Which indexes each thread takes relies on these parameters:
- `blockIdx.x` --> Index of the block (in the x-direction)
- `threadIdx.x` --> Index of the thread in a block (in the x-direction)
- `blockDim.x` --> Number of threads per block (in the x-direction)
- `gridDim.x` --> Number of blocks (in the x-direction)

For my problem, the input is not integer arrays, but a string of `(` and `)`, that I'll denote as `instructions`.  The idea I had was that each thread could take some of the instructions and sum those up (`+1` for `(`, `-1` for `)`). The core device code becomes<sup id="a7">[7](#f7)</sup>
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

Putting it together, where `sum` is the sum from each thread:
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

### Part 2

The problem was to find the first place the partial sum of `instructions` was negative (given the same `+1` and `-1` for `(` and `)`).  I skipped writing a solution in CUDA because it seemed too serial of a problem.

## [Day 2: I Was Told There Would Be No Math](https://adventofcode.com/2015/day/2)

### Part 1

The problem was, given the dimensions of a number of some boxes (right rectangular prisms), find the total sum of the surface areas of all the boxes plus the area of the smallest side from each box.

The solution code for this problem turned out very similar to Day 1. However, while the input for Day 1 had a single line string to parse, Day 2's input was a list of box dimensions, one per line. To handle that, I wrote a (host) function to parse line and save into a vector of strings:

```c++
// Read the data from the input file into a vector of strings, one line per element
std::vector<std::string> data_lines(const std::string& filename)
{
    std::vector<std::string> dataLines;
    std::ifstream data(filename);
    std::string line;
    while(!data.eof())
    {
        std::getline(data, line);
        dataLines.push_back(line);
    }
    return dataLines;
}
```
I ended up using this quite often for the other days to parse the input data lines.

Second, I need those strings converted to integer values to calculate areas and volumes. I ended up with a `int64_t` array of length 3 times the number of boxes: length, width, and height for each boxes.  From there, the device code looks very similar to Day 1:

```c++
    // Paper for this thread 
    int64_t sumPaper = 0;
    // Loop over some boxes (different ones for each thread)
    for (int64_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < N; tid += 3 * blockDim.x * gridDim.x)
    {
        // 3 values for dimensions
        int64_t length = dimensions[3 * tid];
        int64_t width = dimensions[3 * tid + 1];
        int64_t height = dimensions[3 * tid + 2];

        // Paper needed for Part 1 : surface area + smallest side
        sumPaper += 2 * (length * width + width * height + length * height);
        // min3 is just a function to calculate the minimum of 3 values
        sumPaper += min3(length*width, width*height, length*height);
    }

    // Block shared memory array
    __shared__ int64_t paperCache[THREADS];
    const int64_t cacheIndex = threadIdx.x;
    paperCache[cacheIndex] = sumPaper;
    
    __syncthreads();

    reduction(paperCache, cacheIndex);

    if (cacheIndex == 0)
    {
        papers[blockIdx.x] = paperCache[0];
    }
```

Again, this supplies the amount paper needed calculate for each block and then I used `std::accumulate` to calculate the paper needed total across all of the blocks.

### Part 2

Similar to part 1, the problem was to instead find the total sum of the smallest perimeters around each box plus the volume of each box.

The solution looks almost identical to part 1 as well<sup id="a8">[8](#f8)</sup>. The only real change is calculate the smallest perimeter and volume instead of the surface area and smallest side:

```c++
// Ribbon needed for Part 2 : smallest perimeter + volume
sumRibbon += 2 * min3(length + width, width + height, length + height);
sumRibbon += length * width * height;
```

## [Day 4: The Ideal Stocking Stuffer](https://adventofcode.com/2015/day/4)

### Part 1

The problem was to find the smallest integer where the MD5 hash of appending the integer to the given input secret key started with at least 5 zeros.

The trickiest part of solving this problem was getting a MD5 hash algorithm that could run as device code. Of course, plenty of code exists that implements MD5, but I couldn't get any of it to link via `nvcc`, so I ended up implementing my own in [`md5_device.hh`](https://github.com/bobowedge/advent-of-code-2015/blob/main/cuda/md5_device.hh).<sup id="a9">[9](#f9)</sup>

This problem was also the first (but not the last) time that I really lamented not having access to the C++ STL. This meant that I had to implement my own `itoa()` routine to convert an integer to a C-string<sup id="a10">[10](#f10)</sup> and my own routine for concatenating the secret key with an integer:

```c++
// Convert an integer to a C-string
__device__ void itoa(int32_t N, char* Nchar, size_t Nsize)
{
    for (int i = Nsize - 1; i >=0; --i)
    {
        Nchar[i] = (N % 10) + '0';
        N /= 10;
    }
    Nchar[Nsize] = '\0';
}

// Concatenate key and Nchar together
__device__ void concat(char* str, const char* key, unsigned int keyLength, 
                       const char* Nchar, unsigned int Nsize)
{
    int i = 0;
    for (i = 0; i < keyLength; ++i)
    {
        str[i] = key[i];
    }
    for (int j = 0; j < Nsize; ++j, ++i)
    {
        str[i] = Nchar[j];
    }
    str[i] = '\0';
    return;
}
```

After all of those were written, the solution was pretty straightforward: for each `N`, convert to it a string, concatenate to the key and take the MD5 hash. Like in the code for Days 1 and 2, each thread is responsible for different integers:

```c++
unsigned int N = threadIdx.x + blockIdx.x * blockDim.x;
unsigned int increment = blockDim.x * gridDim.x;
while(N < solution)
{
    // Size of N as string
    Nsize = (size_t)log10((double)N) + 1;
    // Convert N to string
    itoa(N, Nchar, Nsize);
    // Concatenate SECRET_KEY and N
    concat(str, key, keyLength, Nchar, Nsize);
    // Compute MD5 of concatenation
    MD5(str, hash, keyLength + Nsize);
    // Check for 5 zeros: 00 00 0X
    if (hash[0] == 0 && hash[1] == 0 && hash[2] < 16)
    {
        atomicMin(&solution, N);
    }
    N += increment;
}
```

As you can see, one of the new CUDA concepts I incorporated was `atomicMin`. This is an atomic operation on shared memory; that is, it blocks other threads and blocks from modifying the value until it's done storing the minimum of the current value of `solution` and `N` back into solution.

In this case, `solution` is marked as `__managed__`, meaning it's in the *global* memory shared by both the host and device, so I can access it in both. In this case, I calculate its value in the device and print it in the host. (This is contrasted with the `__shared__` tag from Days 1 and 2, which is shared *block* memory.)

Additionally, if you look through the github code, you can see I used `cudaDeviceSynchronize()` for the first time, which tells the host to wait until all the blocks reach that point. While I probably should have used it previously, tt didn't matter before for Days 1 and 2. However, this time it actually made a difference: `solution` doesn't return correctly without this synchronization.

### Part 2

Part 2 of the problem upped the difficulty to find the smallest integer where the MD5 hash of appending the integer to the given input secret key started with at least *6* zeros. 

This required essentially a single line change in the core device loop:
```c++
// Check for 6 zeros: 00 00 00
if (hash[0] == 0 && hash[1] == 0 && hash[2] == 0)
{
    atomicMin(&solution, N);
}
```

### Timing

In contrast to Days 1 and 2, this problem definitely benefited from being solved in parallel in CUDA. My Python solution (which is run entirely serially) takes about 15 seconds to generate both solutions. The corresponding CUDA program runs in less than a second; it probably takes longer to print the solution than to calculate it. :thumbsup:

## [Day 6: Probably a Fire Hazard](https://adventofcode.com/2015/day/6)

### Part 1

For Day 6, there is a 1000x1000 grid of lights that need to be turned off, turned on, or toggled based on a series of instructions. After all the instructions are applied, the question is "how many lights are on?"

Because this problem had an inherent 2D grid structure, I decided to use the dimensional blocks and threads, rather than the integer versions this time
```c++
// Block dimensions
const dim3 BLOCKS(8, 16, 1);
// Thread dimensions
const dim3 THREADS(16, 32, 1);
```
This gives a 8x16 grid of compute blocks, where each block has its own rectangle of 16x32 threads.<sup id="a11">[11](#f11)</sup>

The other new concept I used was one of special integer array types available, `int4`, which is an array of 4 integers. This type is what I used to hold the grid points (x1, y1, x2, y2) given in each instruction. Using this type was much easier than allocating and passing an array. 
```c++
int4 grid = make_int4(0, 0, 0, 0);
...
int x1 = grid.x;
int y1 = grid.y;
int x2 = grid.z;
int y2 = grid.w;
```

The approach that I used was to send each instruction from the host to the device in succession. The host code for that:
```c++
// Read the data
auto dataLines = data_lines("../data/day06.input.txt");
// Loop over the instructions
for (auto line : dataLines)
{
    // Parse the instruction from the line
    std::pair<int, int4> typeGridPair = parse_line(line);
    int type = typeGridPair.first;
    int4 grid = typeGridPair.second;
    // Apply the instruction on the device
    apply_instruction<<<BLOCKS,THREADS>>>(grid, type);
}
cudaDeviceSynchronize();
```

For the device code, I needed a way to assign each light in the 1000x1000 grid to a particular block and particular thread within that block. The easiest approach<sup id="a12">[12](#f12)</sup> I found was to treat the row and column each light was in independently. Then, the structure for assigning each light's row and column looks very similar to what we had in the 1D blocks and threads:
```c++
for(int row = threadIdx.y + blockIdx.y * blockDim.y; row < 1000; row += gridDim.y * blockDim.y)
...
for(int col = threadIdx.x + blockIdx.x * blockDim.x; col < 1000; col += gridDim.x * blockDim.x)
...
```
One way to think about this is that each block is responsible for a `THREADS`-sized section of the light grid (and might be responible for more than such section).  The blocks are then spread across the light grid to cover it.

With those assignments in place, the core device code is:
```c++
__global__ void apply_instruction(int type, int4 grid)
{
    const int firstRow = grid.x;
    const int lastRow = grid.z;
    const int firstColumn = grid.y;
    const int lastColumn = grid.w;
    // Each row in the light grid 
    for(int row = threadIdx.y + blockIdx.y * gridDim.y; 
        row < 1000; 
        row += gridDim.y * blockDim.y)
    {
        // Check if this row is in the instruction
        if (row < firstRow || row > lastRow)
        {
            continue;
        }
        // Each col in the light grid
        for(int column = threadIdx.x + blockIdx.x * gridDim.x; 
                column < 1000; 
                column += gridDim.x * blockDim.x)
        {
            // Check if this column is in the instruction
            if (column < firstColumn || column > lastColumn)
            {
                continue;
            }
            //// Apply instruction
            // Toggle
            if (type == 0)
            {
                lights[row][column] ^= 1;
            }
            // Turn off
            else if (type == -1)
            {
                lights[row][column] = 0;
            }
            // Turn on
            else
            {
                lights[row][column] = 1;
            }
        }
    }
}
```

Combined with the previous host code, that applies all of the instructions in order to the light grid. The last thing to do is count the lights that are on. That should be fast on whether its on the host or the device, but I did it on the device. I used the same grid assignment as above to loop over the light grid. I also used the `atomicAdd` function to combine the values from each thread into a global value (`numLightsOn`) that the host could print.
```c++
__device__ __managed__ int numLightsOn = 0;
__global__ void count_lights()
{
    // Count for this thread
    int count = 0;
    for(int row = threadIdx.y + blockIdx.y * blockDim.y; 
        row < 1000; 
        row += gridDim.y * blockDim.y)
    {
        for(int col = threadIdx.x + blockIdx.x * blockDim.x; 
            col < 1000; 
            col += gridDim.x * blockDim.x)
        {
            count += lights[row][col];
        }
    }
    // Add to global count
    atomicAdd(&numLightsOn, count);
}
```

### Part 2

For the second part, each instruction was changed to be an increment or decrement of the light's value rather than a binary operation. The final answer was the sum of the light's values.  Effectively, this only changed the application of instruction in the interior of the device code loops:
```c++
//// Apply instruction
// Toggle
if (type == 0)
{
    lights[row][column] += 2;
}
// Turn off
else if (type == -1)
{
    if (lights[row][column] > 0)
    {
        --lights[row][column];
    }
}
// Turn on
else
{
    ++lights[row][column];
}
```
Everything else remained the same.

### Timing

As in Day 4, the solution to this problem definitely benefited from being done in parallel. My python solution took about 10 seconds, while the executable generated from the CUDA solution prints the solution immediately.

## [Day 9: All in a Single Night](https://adventofcode.com/2015/day/9)

### Part 1

This problem was a take on the traveling salesman problem: given 8 locations and the distance between each pair of location, find the shortest route that visit each location exactly once.

Very naively, there are `8^8 = 16777216` routes that are 8 locations long.<sup id="a13">[13](#f13)</sup> Most of those routes are invalid; they visit a location more than once. However, they are easy to enumerate via repeated division and modular arithmetic. This enumeration allows each route to be mapped to an integer, which can be assigned to a thread, which can them do the determination of whether the route is valid and how long it is.

I took this opportunity to write my first C++ class designed to be used by device code: `Route`. 
Its constructor is the enumeration via division and modular arithmetic (takes integer, sets the destination list). It has two additional functions: `bool valid()`, which checks if a location is listed twice, and `int distance()`, which calculates the distance for the route.<sup id="a14">[14](#f14)</sup>
```c++
// Class to hold a particular route
class Route
{
public:
    // Ordered destination list
    int route[8];

    // Convert an integer to a particular route (enumeration)
    __device__ Route(int N);

    // Check if the route is valid (visit each location exactly once)
    __device__ bool valid() const;

    // Calculate the distance for this route
    __device__ int distance() const;
};
```
With this class in hand, the device code to calculate the shortest route is simple:
```c++
for (int rid = threadIdx.x + blockIdx.x * blockDim.x; rid < 16777216; rid += blockDim.x * gridDim.x)
{
    // Enumerate route
    Route route(rid);
    // Check if route visit each location exacly once
    if (route.valid())
    {
        int distance = route.distance();
        atomicMin(&minDistance, distance);
    }
}
```
(`minDistance` is in global managed memory and initialized to `INT_MAX`.)

### Part 2

Part 2 flips the problem to calculate the longest route. The obvious trivial change works: change to using `atomicMax()` instead of `atomicMin()` after calculating the route's distance, with the global `maxDistance` initialized to 0.

### Timing

My original Python code took about 9-10 seconds to enumerate the routes and find the minimum distance, but that's because I used `itertools.product` to enumerate all `16777216` routes. When I swapped to using `itertools.permutation` (40320 valid routes), it complete immediately.  In any case, the CUDA solution once again runs almost instanteneously.


## [Day 10: Elves Look, Elves Say](https://adventofcode.com/2015/day/10)

The problem was to find the 40th (Part 1) and 50th (Part 2) iterations of a 
[look-and-say sequence](https://en.wikipedia.org/wiki/Look-and-say_sequence), starting with some given input. I couldn't find a parallel way to approach this problem (and there probably isn't one), so I skipped trying to come up with a "better" CUDA version of the solution.

Here's a meme to distract you from the lack of code for this problem:

![Programming Pain](images/ross_meme.jpg){:class="img-responsive"}

## [Day 11: Corporate Policy](https://adventofcode.com/2015/day/11)

### Part 1

This problem was, given the input string of 8 lowercase English letters, find the next string in alphabetical order that meets a precise set of rules, with the premise being that the rules reprsent password policies.

Once again, the trick is converting the problem to something that can be enumerated by integers, so that each enumerated integer can be assigned to a particular thread. In this case, I considered each 
string of 8 letters in alphabetical order:
- "aaaaaaaa" is 0
- "aaaaaaab" is 1
- "aaaaaaac" is 2
- ...
- "zzzzzzzz" is 208827064575 (26^8 - 1)

Each such string maps uniquely to a particular integer.<sup id="a15">[15](#f15)</sup> Denoting `int_to_pwd` (map integer to string) and `pwd_to_int` (map string to int) as functions to handle that mapping, the device code becomes:
```c++
// Find the next valid password
__global__ void next_valid_pwd(int64_t intPwd)
{
    // Buffer for storing password string
    char pwd[PWDLEN];
    for (int64_t N = intPwd + threadIdx.x + blockIdx.x * blockDim.x; N < soln; N +=  blockDim.x * gridDim.x)
    {
        // Convert password integer to string
        int_to_pwd(N, pwd);
        // Check validity of password (password rules)
        if (is_valid(pwd))
        {
            // Set solution
            atomicMin(&soln, N);
            break;
        }
    }
}
```
`intPwd` is set in the host code to correspond to the given input. `soln` (global shared value) is intialized to `208827064576` and becomes the integer corresponding to the next valid password string when the device code is done. Then, the host code can map `soln` back to a string.

### Part 2

The second part of the problem is to find the next valid password after the one found in part 1, so I just run the code back after setting the `intPwd` to be one more than the `soln` found in part 1.

### Timing

Again, the CUDA solution runs incredibly fast with near immediate printing. For the serial Python solution using the same concept, the first part takes about 3 seconds and the second takes about 4 seconds after that.


## [Day 18: Like a GIF For Your Yard](https://adventofcode.com/2015/day/18)

### Part 1

Day 18 marks a return to the light grids from Day 6. The grid is now just 100x100 and the problem is to "animate" the grid. For each step, each light turns off or on based on how many of its neighbors are lit. 100 steps are taken from a given intial configuration.

The first idea I had was to set up a kernel where each thread was responsible for a single light. It could check what the status would be for the next light and then update it, after all the threads synced after doing the same check.  Unfortunately, this doesn't quite work because the total number of threads on a single block is limited to 1024, regardless of shape. A 100x100 = 10,000 thread block is a non-starter with current CUDA standards. 

The second idea I had was to do a 100 blocks of 100 threads.<sup id="a16">[16](#f16)</sup> The issue is that there's no reasonable way to sync the blocks, so you can start writing the new values for the lights before another block has finished evaluating the number of neighbors that are on.

In the end, the approach that I settled on was to have 100 threads. Each thread was responsible for a row of the grid. A thread checks the value for the next step (based on the current neighboring values) for each column in its assigned row, stores that array of values and then reassigns the values after all the threads have synced.  If `next_value(x, y)` gives the value of the light at position `(x,y)` at the next step, the core device loop for evaluating and assigning a single step might look like:
```c++
// Take one step for lights
__global__ void one_step()
{
    // Each thread is responsible for a row
    const int row = threadIdx.x;
    // New values for each column in this row
    int newValues[100];
    // Get next values
    for(int col = 0; col < 100; ++col)
    {
        newValues[col] = next_value(row, col, 1);
    }
    // Sync to make sure not to overwrite values
    __syncthreads();
    // Set new values
    for(int col = 0; col < 100; ++col)
    {
        lights[row][col] = newValues[col];
    }
    // Sync again before moving on to next step
    __syncthreads();
}
```

In the host code, I loop over the steps using this kernel:
```c++
// Loop over the steps and "animate"
for (int step = 0; step < 100; ++step)
{
    one_step<<<1,100>>>();
}
```

From there, I basically reused the counting lights code I used on Day 6 to count the number of lights that are on after 100 steps and get the solution.

### Part 2

The wrinkle in Part 2 was that the corners remained "on" after every step, instead of following the neighbor rules. The only part of my code that is adjusted is the setting new values in `one_step()` to force the value of the lights to be 1.

### Timing

I never wrote the corresponding Python code for Day 18 (or any future days), so there's no comparison to make, but the CUDA code runs basically instanteneously.

## [Day 19: Medicine for Rudolph](https://adventofcode.com/2015/day/19)

### Part 1

For Part 1, the problem was, given a set of string replacements (e.g. "Al" -> "ThF") and a starting string, find the number of unique strings produced by applying a single replacement to the starting string (across all replacements and all locations for each replacement).

The first order was to modify the transformations and input string to be integer-based rather than string-based to make things more efficient. If you're really curious, that's in the `Transforms.hh` file on github. The main thing was to turn the replacement transformations into an array of integer arrays that can be indexed into (43 transforms in all).

The guise of the problem was that each component of the string represented a molecule or element that was being transformed to other molecules, so I will also use that vernacular. In particular, I made a C++ class called `Molecule` to help structure some things:
```c++
struct Molecule
{
    // Represent string of elements (284 is longest possible molecule for this problem)
    int8_t molecule[285];
    // Size of molecule
    size_t msize;
    // Number of steps taken to produce this molecule (used primarily in Part 2)
    size_t steps;

    // Produce an invalid molecule (host and device)
    __device__ __host__ Molecule()
    {
        msize = 285;
        steps = INT_MAX;
    }
    
    // Produce a valid molecule from a string of elements
    __device__ __host__ Molecule(int8_t* newm, size_t newSize, size_t newSteps);

    /** 
     * Produces a new molecule by applying the transform indicated by `transformIdx` 
     * to this molecule. `childNum` indicates how many replacements to skip first 
     * before applying the replacement. An invalid molecule is produced if such a replacement 
     * isn't possible.
     */
    __device__ Molecule fabricate(int64_t transformIdx, int64_t childNum) const;

    // Is this a valid molecule?
    __device__ bool is_valid() const
    {
        return (steps != INT_MAX && msize < 285);
    }
}

// Equality comparison (m1 == m2 if molecules are equal)
__device__ bool operator==(const Molecule& m1, const Molecule& m2);
```

With this `struct` set up, the first device kernel can calculate all of the possible molecules that can be produced from the input molecule:
```c++
// List of current molecules
__device__ __managed__ Molecule* heap;
// Current size of heap
__device__ __managed__ size_t heapSize = 0;

// Make the possible children of `start` and add them to `heap`
__global__ void fabricate_molecules(Molecule* start)
{
    for (int64_t transformIdx = threadIdx.x; 
                 transformIdx < 43; 
                 transformIdx += blockDim.x)
    {
        int64_t childNum = 0;
        // Produce first child
        Molecule m = start[0].fabricate(transformIdx, childNum);
        // Continue we produce an invalid child or run out of space
        while(m.is_valid() && heapSize < MAX_HEAP_SIZE)
        {
            // Add to heap
            int64_t heapIdx = atomicAdd(&heapSize, (size_t)1);
            heap[heapIdx] = m;

            ++childNum;
            
            // Produce next child
            m = start[0].fabricate(transformIdx, childNum);
        }
    }
}
```
`fabricate_molecules` produces all of the possible children from applying a single transformation to the input molecule. The next step is to count the unique strings produced, as the fabrication can produce some duplicates. That's mostly handled in the same way I've been counting things in the previous problems, using the `reduction` routine to thread the counting. The `operator==` defined for `Molecule` is used to check for duplicates while each thread is going through the heap.

<a name="atomicAddTrick"></a> I wanted to highlight the bit of code around `heapIdx` and `atomicAdd` since I thought it was particularly slick. (I didn't come up with it.) The `atomicAdd` increments the heapSize counter and returns the previous value of heapSize. It allows you to insert a new object into the heap without worrying about other threads or blocks conflicting, because they get the next increment (assuming `atomicAdd` is called in the same way before their insert).

### Part 2

The problem for part 2 was to find the smallest number of steps required to produce the given input string using the transformations given (same ones discussed in part 1), starting from the string 'e' (representing a single electron in the molecule parlance).  Each applied transformation counted as a step.

My approach was to start with the input string and repeatedly 'reverse apply' transformations until the 'e' was reached.<sup id="a17">[17](#f17)</sup> The replacements are such that each reverse application will always reduce the size of the molecule. 
> For example, if the original replacement was `Al => ThF`, then one possible step would be to replace a single instance of `ThF` with `Al`. 

Each such replacement can be consider as a branch of a tree with the original molecule as the root node. The first part of the problem then is to find a path through the tree, i.e. a series of reverse transformations, that terminates with 'e'. Given that I'm working in the parallel world of CUDA, my plan was to find the shortest branch by exhausting the possible paths over some part of the tree and eliminating the remaining paths as solutions because they were as long or longer.

The procedure I used was as follows:
1. Apply reverse transformations to element(s) at the top of the heap and put the valid ones on the heap
2. Sort the heap, first by size, then by number of steps taken.
3. Continue 1-2 until 'e' is reached.
4. Process remaining paths until they reach 'e' or they are as long as the best path so far.

For the host code, this looks like
```c++
// Continue until done or memory gets full
while (heapSize > 0 && heapSize < MAX_HEAP_SIZE)
{
    // Apply each reverse transformation to the top of the heap
    deconstruct_molecules<<<1,THREADS>>>();
    cudaDeviceSynchronize();
    // Sort the heap
    bitonic_sort();
    // Recalculate the heap size after the sort
    reset_heap_size<<<1,1>>>();
    cudaDeviceSynchronize();
}
```

A [bitonic_sort](https://en.wikipedia.org/wiki/Bitonic_sorter) was a new concept to me. It is a sorting algorithm that can be implemented in parallel code, so I did it: <a name="bitonicSort"></a>
```c++
// Single step of the bitonic_sort()
__global__ void bitonic_sort_step(int64_t j, int64_t k)
{
    // Each thread handles a different element
    int64_t i = threadIdx.x + blockDim.x * blockIdx.x;
    int64_t ij = i ^ j;
    if (ij > i)
    {
        // Determines whether or not to swap indices
        int64_t ik = i&k;
        if ( (ik == 0 && heap[ij] < heap[i]) ||
             (ik != 0 && heap[i] < heap[ij]) )
        {
            Molecule temp = heap[i];
            heap[i] = heap[ij];
            heap[ij] = temp;
        }
    }
}

// Parallel sort algorithm
__host__ void bitonic_sort()
{
    for (int64_t k = 2; k <= MAX_HEAP_SIZE; k <<= 1)
    {
        for(int64_t j = k >> 1; j > 0; j = j >> 1)
        {
            bitonic_sort_step<<<BLOCKS,THREADS>>>(j, k);
        }
    }
    cudaDeviceSynchronize();
}
```
You may notice that `bitonic_sort_step()` requires the comparison of two heap objects, in this case, `Molecule` objects, so we need a less than comparison operator:
```c++
__device__ bool operator<(const Molecule& m1, const Molecule& m2)
{
    if (m1.msize != m2.msize)
    {
        return m1.msize < m2.msize;
    }
    return m1.steps < m2.steps;
}
```
This sort places the invalid molecules at the end of the heap and the shortest valid molecule at the top of the heap.

The next bit of device code is a way to the apply the reverse transformations to an individual `Molecule`:
```c++
// Convert a molecule to a smaller one via a particular reverse transformation
__device__ Molecule Molecule::deconstruct(int64_t transformIdx) const;
}
```
`transformIdx` again indicates the particular transformation to use, this time in reverse. As before, an invalid `Molecule` is produced if the transformation is not possible.

Using this `deconstruct` function to apply a reverse transformation to molecule, this is what the device code for `deconstruct_molecules()` looks like:
```c++
// Apply the reverse transformations to the top of the heap
__global__ void deconstruct_molecules()
{
    if (blockIdx.x < heapSize)
    {
        // Each block grabs a different Molecule from the top of the heap
        Molecule start = heap[blockIdx.x];
        // Each thread applies a different reverse transformation
        for(int64_t transformIdx = threadIdx.x; transformIdx < 43; transformIdx += blockDim.x)
        {
            // Create the new molecule
            Molecule m = start.deconstruct(transformIdx);
            if (!m.is_valid())
            {
                continue;
            }
            // Reached 'e'
            if (m.msize == 1 && m.molecule[0] == 16)
            {
                atomicMin(&bestSteps, m.steps);
            }
            else if (m.steps < bestSteps)
            {
                // Add new molecule to the end of the heap
                int64_t heapIdx = atomicAdd(&heapSize, static_cast<size_t>(1));
                heap[heapIdx] = m;
            }
        }
        // This Molecule has been processed, so make it invalid
        if (threadIdx.x == 0)
        {
            heap[blockIdx.x] = Molecule();
        }
    }
}
```
The code is now set up to run through the tree to find a series of transformations that reduces the input molecule to 'e'...Or not.

Turns out, this doesn't actually work. More accurately, it works to find *a* path but it's not going to exhaust enough paths to finish in my lifetime. There are on the order of 195<sup>23</sup> paths to check. :grimacing:

Some googles and a bit of reading later,<sup id="a18">[18](#f18)</sup> I learned that the 'true' solution relies on the fact that particular structure of the input molecule guarantees that any path that terminates with 'e' with have the same length. For the general case, this guarantee is not valid.

Given that added guarantee, I know that if I find one path, it's the best one. My workaround is to modify the `while` condition in the host code slightly:
```c++
while (heapSize > 0 && heapSize < MAX_HEAP_SIZE) && bestSteps == INT_MAX)
{...
```
Now, when it finds a path, the loop ends and `bestSteps` is returned. (With the 'correct' answer.)

### Timing

The solution to Part 1 runs almost instantly. The solution for Part 2 runs in less than half a second with the `while` short circuit above.  Not quite instantly, but still satisfying.

## [Day 20: Infinite Elves and Infinite Houses](https://adventofcode.com/2015/day/20)

### Part 1

The premise for this problem is that Santa is having elves deliver presents door-to-door to each house, with each elf delivering ten times its assigned number to each multiple of its number. (Elf 1 delivers 10 presents to all houses; Elf 2 delivers 20 present to each even house; Elf 3 delivers 30 presents to each house number divisible by three, and so on.)  The question is what is the smallest house number that gets above a certain given number of presents. 

I came up with two approaches to this problem and ended up implementing them both to see which was faster. 

The first approach was to go house by house, figure out which elves delivered there (i.e. the factors of the house number), and add up the presents for that house. The solution arrives at the first house that is above the threshold.
```c++
// Given number of input presents
const uint32_t PUZZLE_INPUT = 36000000;
// Maximum house number (guaranteed to be no bigger than this)
const uint32_t MAX_HOUSES = PUZZLE_INPUT/10;
// House stack 
__device__ __managed__ uint32_t* houses;
// Deliver the presents house by house
__global__ void deliver_by_house()
{
    int32_t presents = 0;
    // Loop over the houses (starting at 1)
    for(uint32_t house = 1 + threadIdx.x + blockIdx.x * blockDim.x;
        house < MAX_HOUSES;
        house += gridDim.x * blockDim.x)
    {
        // Loop over the possible elfs for this house
        for (uint32_t elf = 1; elf <= house; ++elf)
        {
            // Does this elf deliver here (i.e. is elf a factor of house)?
            if ((house % elf) == 0)
            {
                presents = 10 * elf;
                if (houses[house] + presents >= PUZZLE_INPUT)
                {
                    atomicExch(&houses[house], PUZZLE_INPUT);
                }
                else
                {
                    atomicAdd(&houses[house], presents);
                }
            }
        }
    }
}
```
The only new part in here is `atomicExch`, which atomically sets the value. I use it here to set the house value to the puzzle input if the house value would exceed that. This speeds up the code a little as it prevent unnecessary adds once the value is exceeded.<sup id="a19">[19](#f19)</sup> From here, it's easy to loop over the houses and find the smallest house with value greater than or equal to `PUZZLE_INPUT`.

The second approach I came up with was to deliver the presents elf by elf. So, each elf delivers all its presents to all its houses before going to the next elf.
```c++
// Deliver the presents elf by elf
__global__ void deliver_by_elf()
{
    uint32_t presents = 0;
    // Loop over elves (starting at 1)
    for(uint32_t elf = 1 + threadIdx.x + blockIdx.x * blockDim.x;
        elf < MAX_HOUSES;
        elf += gridDim.x * blockDim.x)
    {
        presents = 10 * elf;
        // Presents for this elf
        for (uint32_t j = elf; j < MAX_HOUSES; j += elf)
        {
            if (houses[j] + presents >= PUZZLE_INPUT)
            {
                atomicExch(&houses[j], PUZZLE_INPUT);
            }
            else
            {
                atomicAdd(&houses[j], presents);
            }
        }
    }
}
```
Again, this uses `atomicExch` to set the value to prevent unneeded adds. 

This approach counting elf-by-elf turned out to be much, much faster than the house-by-house approach. See more in the [Timing](#timing-6) section below.

### Part 2

Part 2 modified the problem by only allowing each elf to deliver to 50 houses, i.e. the first 50 multiples of its number, and increasing the number of presents it delivered to be 11 times its number.  This doesn't change much of the code, it adds the verification that, for each house value, `house <= 50 * elf` before adding new presents.

### Timing

For this problem, I decided to interrogate how the number of 'blocks' and 'threads' influenced how long it took to solve the problem (in wall time).  The solution to both parts could be calculated  simultaneously, so I ran some timing code for the two different approaches (house-by-house and elf-by-elf) that I used to solve the problem.

For the house-by-house approach, here's what the timing looked like:

| # of blocks | # of threads | Wall Time |
| ----------- | ------------ | ----------|
|          32 |          128 |   164 sec |
|          64 |          128 |    87 sec |
|         128 |          128 |    57 sec |
|         256 |          128 |    54 sec |
|         512 |          128 |    51 sec |
|        1024 |          128 |    47 sec |
|        2048 |          128 |    46 sec |
|          32 |          256 |    93 sec |
|          64 |          256 |    56 sec |
|         128 |          256 |    53 sec |
|         256 |          256 |    51 sec |
|         512 |          256 |    48 sec |
|        1024 |          256 |    47 sec |
|        2048 |          256 |    46 sec |

Obviously, more blocks or more threads is faster, but the speed gains are limited after a point.

For the elf-by-elf approach, here's the timing table:

| # of blocks | # of threads | Wall Time  |
| ----------- | ------------ | ---------  |
|          32 |          128 |   0.91 sec |
|          64 |          128 |   0.89 sec |
|         128 |          128 |   0.90 sec |
|         256 |          128 |   0.90 sec |
|         512 |          128 |   0.90 sec |
|        1024 |          128 |   0.90 sec |
|          32 |          256 |   0.89 sec |
|          64 |          256 |   0.90 sec |
|         128 |          256 |   0.90 sec |
|         256 |          256 |   0.90 sec |
|         512 |          256 |   0.90 sec |
|        1024 |          256 |   0.90 sec |

These timings are all essentially identical. I also tried some smaller numbers for this case:

| # of blocks | # of threads | Wall Time  |
| ----------- | ------------ | ---------  |
|          64 |           32 |   0.92 sec |
|          64 |           16 |   0.89 sec |
|          64 |            8 |   0.89 sec |
|          32 |           16 |   0.94 sec |
|          16 |           16 |   0.96 sec |
|           8 |            8 |   1.14 sec |

I don't have anything groundbreaking to say about how many blocks or threads to use and the timing; I just wanted to see what it looked like if I changed them.

## [Day 21: RPG Simulator 20XX](https://adventofcode.com/2015/day/21)

### Part 1

The premise for this problem is that an RPG player finds a shop just before the boss fight. The shop sells various items to set armor and attack damage and the goal is to find the smallest cost the player can pay and still win the boss fight. (The player starting HP and the boss' starting HP, damage, and armor are all fixed inputs.)

The rules also stipulate exactly one weapon (out of 5), 0-1 armor (out of 5), and 0-2 rings (out of 6), which gives `5*6*22 = 660` possible equipment combinations.  These can be easily enumerated to assign each number < 660 to a unique combination. Thus, the overall procedure, for each number will be:
1. Given number N, calculate the cost, armor, and attack damage from the corresponding equipment.
2. Determine who wins the fight if the player dons that equipment
3. If the player wins, compare the current cost to the best cost so far and set the best cost appropriately.

For the device code, I implemented this as
```c++
// Cost (746 = buy everything)
__device__ __managed__ int bestCost = 746;
// Find best cost
__global__ void find_costs()
{
    for(int N = threadIdx.x; N < 660; N += blockDim.x)
    {
        int playerCost = 0;
        int playerDamage = 0;
        int playerArmor = 0;
        // Convert N to equipment
        calculate_equipment(N, playerCost, playerDamage, playerArmor);
        // Determine fight winner
        bool winner = win_fight(playerDamage, playerArmor);
        // Set best cost
        if (winner && playerCost < bestCost)
        {
            atomicMin(&bestCost, playerCost);
        }
    }
}
```

### Part 2

The modification for this part was to find the maximum the player could spend at the shop (with the same equipment limitations) and still lose the boss fight. The code change is just to swap the conditional for setting the best cost to set the worst cost instead:

```c++
// Cost (buy nothing)
__device__ __managed__ int worstCost = 0;
...
// Set worst cost
if (!winner && playerCost > worstCost)
{
    atomicMax(&worstCost, playerCost);
}
...
```

### Timing

This is decidedly not a problem that requires a parallel solution. There are only 660 possible combinations to check, so a single thread runs almost instantly, just as the 660 thread solution does. C'est la vie.

## [Day 22: Wizard Simulator 20XX](https://adventofcode.com/2015/day/22)

### Part 1

The premise for this problem is similar to Day 21: there's an RPG player in a boss fight.  This time, however, the player is a wizard who has set of 5 spells that can be cast on each turn. Each spell costs a different amount of mana to use and can have effects like healing, damaging the boss, increasing armor, or increasing the amount of remaining mana. The problem is to find the least amount of mana that can be spent to win the fight. (Player and boss starting HP, boss damage, and player starting mana pool are all fixed inputs. Boss has zero armor.)

The vision I had for solving this problem was similar to ones I've used above: construct a mechanism for converting from an integer to a sequence of spells. In this case, there is straightforward way to do so.  Each spell is assigned a number from 0-4.  Given an integer,  repeated division and modding by 5 of that integer will give a sequence of numbers 0-4. That sequence of integers corresponding directly to a sequence of spells.

I set up the player and boss as C++ classes to make it a little easier to track each round (see github for omitted details):
```c++
// Boss class
class Boss
{
public:
    // HP (51 is given input)
    int hp = 51;
    // Damage (9 is given input)
    int damage = 9;
    // Constructor
    __device__ Boss() {};
    // Boss attack player for damage
    __device__ void attack(Player& player);
    // Check if the boss is dead (HP <= 0)
    __device__ bool is_dead();
};

// Player class
class Player
{
public:
    // Total mana spent so far
    int manaSpent = 0;
    // Mana pool (500 is given input)
    int mana = 500;
    // HP (50 is given input)
    int hp = 50;
    // Constructor
    __device__ Player() {};
    // Magic missile spell (spell 0)
    __device__ bool magic_missile(Boss& boss);
    // Drain spell (spell 1)
    __device__ bool drain(Boss& boss);
    // Shield spell (spell 2)
    __device__ bool shield()
    // Poison spell (spell 3)
    __device__ bool poison()
    // Recharge spell (spell 4)
    __device__ bool recharge()
    // Check if there's enough mana to cast any spell
    __device__ bool can_cast();
    // Check if player is dead
    __device__ bool is_dead();
};
```

To execute one round (one player turn and one boss turn), then I implemented as
```c++
// Return true if round is valid (player can cast)
__device__ bool round(int64_t& spellList, Player& player, Boss& boss)
{
    // Get next spell
    int spell = spellList % 5;
    // Take player turn (player spell, etc.)
    bool valid = player_turn(spell, player, boss);
    // Invalid if not enough mana to cast spell
    if (!valid)
    {
        return false;
    }
    // Take boss turn (boss attack, etc.)
    boss_turn(player, boss);
    // Discard last spell
    spellList /= 5;
    // Valid round
    return true;
}
```

With those pieces set, I found the best mana by given each thread a different integer:
```c++
// Find the smallest mana to use to beat boss
__global__ void find_best_mana(bool hardMode)
{
    for(int64_t N = threadIdx.x + blockIdx.x * blockDim.x; N < INT_MAX; N += blockDim.x * gridDim.x)
    {
        int64_t spellList = N;
        Player player;
        Boss boss;
        // Continue until boss is dead, player is dead, or too much mana is spent
        while(player.can_cast() && !player.is_dead() && !boss.is_dead() && player.manaSpent < bestMana)
        {
            // Run a single round (player turn + boss turn)
            if (!round(spellList, player, boss, hardMode))
            {
                break;
            }
        }
        // If boss is dead, check against best mana
        if (boss.is_dead())
        {
            atomicMin(&bestMana, player.manaSpent);
        }
    }
}
```

### Part 2

For Part 2, the change was that the player took an additional HP loss at the start of their turn (*hard mode*). This added about 3 lines of code to `player_turn` to account for this.

### Timing

For the code as written and outlayed above, it takes less than one second each for both parts, but it isn't quite instanteous.

## [Day 23: Opening the Turing Lock](https://adventofcode.com/2015/day/23)

This is decidedly not a parallel problem.  As such, I'm skipping any discussion of it here, but you can check out the github repo for `day23.cu`, if you're dying to see some CUDA code solving it.

As recompense, here's a meme:

![NVIDIA Master Plan](../images/NVIDIAs-Master-Plan.jpg){:class="img-responsive"}

And another:

![Captain America NVIDIA](../images/Cap_NVIDIA.jpeg){:class="img-responsive"}

## [Day 24: It Hangs in the Balance](https://adventofcode.com/2015/day/24)

### Part 1

Under the guise of balancing the presents on sleigh, the problem for Day 24 was to divide a group of integers into 3 groups so that the sum of each of the groups was the same. Additionally, you needed to find the arrangement that had the group with the smallest [cardinality](https://en.wikipedia.org/wiki/Cardinality) and, if there were multiple such arrangements, the one with the smallest product within that set with the smallest cardinality.

There were 28 weights to put into the 3 groups. Thus, any group in an arrangement could be represented by some nonnegative integer less than `2**28` by considering the binary representation of the integer as a indicator of whether the weight is in the group.  For example,
> If N = 100663297 = 110000000000000000000000001, the group would have the first weight, the second-to-last weight, and the last weight in it (and nothing else).

1. Find the number of possible smallest groups (by looping over all non-negative integers < 2**28): those groups that sum to 1/3 of the total weight and fewer than 10 elements.
2. Allocate a heap of integers of the size found in step 1.
3. Add each valid integers from step 1 to heap
4. Sort the heap (using `bitonic_sort`), first by cardinality of the corresponding group, then by the product of the elements of the group.
5. For each non-negative integers < 2**28, get a group of weights ("second group"). If this second group sums to 1/3 the weight, test it against every "first group" from the heap. If they don't overlap in any weights, this also gives a "third group" to give a valid arrangement. If so, test the cardinality and product of the "first group" against the best arrangement found so far.

Following these steps will give the desired arrangement.  The device code for each step is laid out below for each step:

1. Find the number of groups that sum to 1/3 of the total weight and have fewer than 10 elements
```c++
// Total sum of WEIGHTS
const uint32_t TOTAL_WEIGHT = 1524;
// 1/3 of TOTAL_WEIGHT
const uint32_t GROUP_WEIGHT = TOTAL_WEIGHT / 3;
// Heap of integers representing sets of weights
__device__ __managed__ uint32_t* heap;
// Size of heap
__device__ __managed__ size_t heapSize = 0;
// Current index of heap
__device__ __managed__ size_t heapIdx = 0;
// Part 1, Step 1
__global__ void find_first_groups()
{
    uint32_t weights[28];
    size_t cardinality = 0;
    uint32_t sumWeights = 0;
    for(uint32_t N = threadIdx.x + blockDim.x * blockIdx.x; N < 268435456// 2**28
        N += blockDim.x * gridDim.x)
    {   
        // Convert integer to weights group
        cardinality = convert_weights(N, weights);
        // If cardinality > 9, it's not the smallest group in arrangement
        if (cardinality > 9)
        {
            continue;
        }
        // Weights must sum to 1/3 TOTAL WEIGHT
        sumWeights = sum(weights);
        if (sumWeights != GROUP_WEIGHT)
        {
            continue;
        }
        // Increment heap size to make room for this N in step 3
        atomicAdd(&heapSize, static_cast<size_t>(1));
    }
}
```

2. No device code for this step: Allocate the first power of 2 larger than `heapSize` number of integers for `heap` (Power of 2 required for `bitonic_sort` in step 4).

3. Repeat step 1, but add the integers to the heap that has now been allocated, so I replaced the `atomicAdd` above with 
```c++
// Add this integer to the heap
size_t idx = atomicAdd(&heapIdx, (size_t)1);
heap[idx] = N;
```
This uses the same incrementing an index trick I used in [Day 19](#atomicAddTrick) to allow each thread to store to the heap without clobbering writes by other threads.

4. Sort the heap by considering the cardinalities and products of the weights each integer in the heap represents. The sorting algorithm is `bitonic_sort`, using basically the same code that I used on [Day 19](#bitonicSort) as well.

5. Lastly, I looped over the possible second group of weights and tested each first group that made a valid arrangement to find the first group with the lowest cardinality (and smallest product within the groups with the same cardinality.)
```c++
// Smallest cardinality found for first group
__device__ __managed__ uint32_t bestCardinality = UINT_MAX;
// Smallest product found in smallest cardinality for first group
__device__ __managed__ uint64_t bestQE = UINT64_MAX;
// Part 1, Step 5
__global__ void test_valid_arrangements()
{
    // Weight arrays
    uint32_t weights1[LENGTH];
    uint32_t weights2[LENGTH];
    // Cardinalities
    uint32_t cardinality1 = 0;
    uint32_t cardinality2 = 0;
    uint32_t cardinality3 = 0;
    // Sum of weights
    uint32_t sumWeights2 = 0;
    // Product of weights (quantum entanglement)
    uint64_t qe1 = 0;
    // Loop over integers < 2**28 (second group)
    for(int64_t N2 = threadIdx.x + blockDim.x * blockIdx.x; 
        N2 < POSSIBLES; 
        N2 += blockDim.x * gridDim.x)
    {   
        // Get second group weights
        cardinality2 = convert_weights(N2, weights2);
        // Calculate sum of weights
        sumWeights2 = sum(weights2);
        // Check for invalid group
        if (sumWeights2 != GROUP_WEIGHT)
        {
            continue;
        }
        // Loop over all first group weights in heap
        for (uint64_t idx = 0; idx < heapSize; ++idx)
        {
            uint32_t N1 = heap[idx];
            // If N1 & N2 is not zero, weights overlap
            if ((N1 & N2) != 0)
            {
                continue;
            }
            // Get first group weights
            cardinality1 = convert_weights(N1, weights1);
            // Check if already bigger than bestCardinality or smaller than second group
            if (cardinality1 > bestCardinality || cardinality2 < cardinality1)
            {
                break;
            }
            // Cardinality of third group
            cardinality3 = LENGTH - (cardinality1 + cardinality2);
            // Check if it's smaller than first group
            if (cardinality3 < cardinality1)
            {
                continue;
            }
            // Test against the best arrangement so far
            qe1 = product(weights1);
            if (cardinality1 == bestCardinality && qe1 >= bestQE)
            {
                break;
            }
            else
            {
                atomicMin(&bestCardinality, cardinality1);
                atomicMin(&bestQE, qe1);
                break;
            }
        }
    }
}
```
This sets `bestCardinality` and `bestQE` to be the desired values.

### Part 2

Part 2 had the same basic premise as Part 1, except now, instead of 3 equally summed groups, there were 4. 

To tackle this problem, I used a similar approach as I did in Part 1 to start, however, it needed to be modified a bit after that to account for the extra group. Steps 1-4 from Part 1 were identical:

1. Find the number of possible smallest groups (by looping over all non-negative integers < 2**28).Those groups that sum to 1/4 of the total weight and fewer than 10 elements.
2. Allocate a heap of integers of the size found in step 1.
3. Add each valid integer from step 1 to heap
4. Sort the heap (using `bitonic_sort`), first by cardinality of the corresponding group, then by the product of the elements of the group.

Next, I made a second heap that corresponded to all other groups that could co-exist with at least one group in the first heap. It plays very similar to Steps 1-3:

{:start="5"}
5. Find the number of possible other groups (by looping over all non-negative integers < 2**28): those groups that sum to 1/4 of the total weight and have at least one non-overlapping group in the first heap whose cardinality is not bigger than its cardinality.
6. Allocate a second heap of integers of the size found in step 5.
7. Add each valid integer from step 5 to the second heap.

The last step in Part 2 is a bit different than the last step in Part 1:

{:start="8"}
8. For every possible pair of groups in the second heap ("second" and "third" groups), find the non-overlapping groups in the first heap ("first" groups). If none of the 3 groups overlap, this also gives a "fourth" group of the remaining unused weights to specify a valid arrangement. If so, test the cardinality and product of the "first" group against the best arrangement found so far. 

In terms of device code, steps 1-4 are basically identical to Part 1. For the remaining steps:

{:start="5"}
5. Find the possible second groups that have at least one valid first group:
```c++
// Heap of integers represeting sets of weights for other groups (Part 2 only)
__device__ __managed__ uint32_t* otherHeap;
// Size of otherHeap
__device__ __managed__ size_t otherHeapSize = 0;
// Part 2, Step 5
__global__ void find_other_groups(bool addToHeap)
{
    // Weight arrays
    uint32_t weights1[LENGTH];
    uint32_t weights2[LENGTH];
    // Cardinalities
    uint32_t cardinality1 = 0;
    uint32_t cardinality2 = 0;
    // Sum of weights
    uint32_t sumWeights2 = 0;
    // Allocate some temp integers
    uint32_t N1 = 0;
    // Loop over integers < 2**28 for second group
    for(uint32_t N2 = threadIdx.x + blockDim.x * blockIdx.x; 
                 N2 < POSSIBLES; 
                 N2 += blockDim.x * gridDim.x)
    {   
        cardinality2 = convert_weights(N2, weights2);
        sumWeights2 = sum(weights2);
        // Check if valid group
        if (sumWeights2 != GROUP_WEIGHT2)
        {
            continue;
        }
        // Check if there's at least one valid first group that goes with this second group
        bool validN2 = false;
        // Loop over all first group weights in heap
        for (uint64_t idx = 0; idx < heapSize; ++idx)
        {
            N1 = heap[idx];
            // If N1 & N2 is not zero, weights overlap
            if ((N1 & N2) != 0)
            {
                continue;
            }
            // Second group can't be smaller than first group
            cardinality1 = convert_weights(N1, weights1);
            if (cardinality2 < cardinality1)
            {
                continue;
            }
            // Found a possible first group, so break
            validN2 = true;
            break;
        }
        // Possible valid N2
        if (validN2)
        {
            // Increment otherHeap size to make room for this N2 in next pass
            atomicAdd(&otherHeapSize, static_cast<size_t>(1));
        }
    }
}
```

6. This is similar to step 2: no device code here, just allocate correct memory for `otherHeap`.

7. Repeat step 5, but add the integers to the heap that has now been allocated, so I replaced the `atomicAdd` line above with 
```c++
// Add this integer to the heap
size_t idx = atomicAdd(&heapIdx, static_cast<size_t>(1));
otherHeap[idx] = N2;
```
(Same as was done in step 3.)

8. Lastly, loop over the second heap twice and the first loop once to find valid arrangements and test the first groups to find the one with the lowest cardinality and product:
```c++
// Part 2, Step 8
__global__ void test_valid_arrangements2()
{
    // Weight arrays
    uint32_t weights1[LENGTH];
    uint32_t weights2[LENGTH];
    uint32_t weights3[LENGTH];
    // Cardinalities
    uint32_t cardinality1 = 0;
    uint32_t cardinality2 = 0;
    uint32_t cardinality3 = 0;
    uint32_t cardinality4 = 0;
    // Product (quantum entanglement)
    uint64_t qe1 = 0;
    // Loop over otherHeap (third group)
    // Each block handles a different third group
    for(int64_t idx3 = blockIdx.x; idx3 < otherHeapSize; idx3 += gridDim.x)
    {   
        uint32_t N3 = otherHeap[idx3];
        cardinality3 = convert_weights(N3, weights3);
        // Loop over otherHeap (second group)
        // Each thread handles a different second group
        for(int64_t idx2 = threadIdx.x; idx2 < otherHeapSize; idx2 += blockDim.x)
        {
            uint32_t N2 = otherHeap[idx2];
            // Check for overlap between groups
            if ((N3 & N2) != 0)
            {
                continue;
            }
            // Cut the search space in half by only considering when cardinality2 <= cardinality3
            cardinality2 = convert_weights(N2, weights2);
            if (cardinality2 > cardinality3)
            {
                continue;
            }
            // Loop over possible first groups
            for (int64_t idx1 = 0; idx1 < heapSize; ++idx1)
            {
                N1 = heap[idx1];
                cardinality1 = convert_weights(N1, weights1);
                // Can't do any better with the rest of the heap
                if (cardinality1 > bestCardinality)
                {
                    break;
                }
                // Group 1 needs to be the smallest
                if (cardinality1 > cardinality2)
                {
                    break;
                }
                // Group 1 needs to be the smallest
                cardinality4 = LENGTH - (cardinality1 + card23);
                if (cardinality1 > cardinality4)
                {
                    break;
                }
                // Test against the best arrangement so far
                qe1 = product(weights1);
                if (cardinality1 == bestCardinality && qe1 >= bestQE)
                {
                    break;
                }
                else
                {
                    atomicMin(&bestCardinality, cardinality1);
                    atomicMin(&bestQE, qe1);
                    break;
                }
            }
        }
    }
}
```
This sets the correct `bestCardinality` and `bestQE` values for Part 2.

### Timing

Despite having several steps and being almost 600 lines of code for both parts for this problem, the code runs basically instantly. Both the heap sizes and the short circuiting of doing further checks have a lot to do with that. For Part 1, the heap size (i.e. the number of valid first groups) was 20679. For Part 2, the first heap had 11179 elements and the second heap (i.e. number of valid "other" groups) had 54966 elements. 

## [Day 25: Let It Snow](https://adventofcode.com/2015/day/25)

Like Day 23, this is definitely not a parallel problem. I wrote the solution in 30 lines of C++ code, though, if you want to look at that. I didn't even bother with pretending to write CUDA for this problem.

<iframe width="560" height="315" src="https://www.youtube.com/embed/gcJjlL3izVc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Conclusion

Thanks to anyone who actually read the entire way down here. I owe you a hearty handshake/hug once this pandemic is over. 

No real conclusions here, just the smug satisfications of writing some parallel code that compiles :open_mouth: and works :boom: :boom: :boom:.

## Footnotes

<b id="f1">1</b>I came close to learning some CUDA before: I took an High Performance Computing class two years ago that covered programming in CUDA for a day or two. Unfortunately, my daughter ran a fever at daycare that day, so I missed it and couldn't find the time to revisit it. Now, however, with the opportunity to learn it at my own pace, I had no excuses.[↩](#a1)

<b id="f2">2</b> At first, because I wanted to get on the leaderboard. However, it became quickly clear  that was never happening, but I kept up the facade of doing it as quickly as *I* could.[↩](#a2)

<b id="f3">3</b> Each day is a 2-part problem, where the second part is revealed after correctly answering the first part[↩](#a3)

<b id="f4">4</b> One of the cool things about Advent of Code is that each person gets a different batch of input data.[↩](#a4)

<b id="f5">5</b> This seems to be the "hello world" equivalent for CUDA. In addition to being the example that NVIDIA uses, it's the first meaningful example in the book I mentioned above,and pretty much all of the video examples if you search for "Introduction to CUDA". [↩](#a5)

<b id="f6">6</b> Opinions are like my friends, they're all assholes. :fire:  That's how the saying goes, right?[↩](#a6)

<b id="f7">7</b> Ternary statements are hot garbage, in case you're wondering.[↩](#a7)

<b id="f8">8</b> If you look at the github code, the paper and ribbons calculations are done in the same loop. [↩](#a8)

<b id="f9">9</b> I don't recommend that route either. I eventually got it to work, but debugging it required compiling and linking another C++ version and lots of print statements to figure out where I had put a `+` instead of a `-`. [↩](#a9)

<b id="f10">10</b> Perhaps more to the point, `itoa()` is not in the ANSI-C standard and not supported by `nvcc`. [↩](#a10)

<b id="f11">11</b> Those values are mostly arbitrary: I used a non-square grid to make sure that I was using the grid correctly, but kept it 2D to avoid unneeded complication. Cue obligatory Khan from Star Trek 2 joke. [↩](#a11)

<b id="f12">12</b> Definitely not the first or second or third approach, though.[↩](#a12)

<b id="f13">13</b> Of course, there are `8! = 40320` routes that visit each location exactly once. There is an [algorithm](https://en.wikipedia.org/wiki/Heap%27s_algorithm) to efficiently enumerate those routes, but I don't know it off-hand, so I took a different tack.[↩](#a13)

<b id="f14">14</b> I'll spare you the function definitions here since they're not particular interesting, but see the github repo if you're compelled to see them.[↩](#a14)

<b id="f15">15</b> `int` will not suffice, `int64_t` is needed.[↩](#a15)

<b id="f16">16</b> This seem to work accidentally on my machine for Part 1, but breaks gloriously for Part 2, giving nondeterminstic answers.[↩](#a16)

<b id="f17">17</b> I actually tried the forward way, too, but with less success.[↩](#a17)

<b id="f18">18</b> I probably should have done this earlier than I did, but I really wanted my code to work as is. :flushed: [↩](#a18)

<b id="f19">19</b> Strictly speaking, the atomics probably aren't needed, but they don't slow down the code enough to notice and feel safer in preventing overwrites. [↩](#a19)

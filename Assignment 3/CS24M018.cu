#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <bits/stdc++.h>

#define MOD 1000000007

using namespace std;

struct Edge
{
    int *src;
    int *dest;
    int *weight;
    int *type;
} Edges, d_Edges;

__device__ int mstWeight;
__device__ const int d_MOD = 1000000007;

// find parent of a vertex
__device__ int findParent(int *parent, int vertex)
{
    while (parent[vertex] != vertex)
    {
        vertex = parent[vertex];
    }
    __threadfence();

    return vertex;
}

// merge 2 components.
__device__ void merge(int *parent, int src, int dest)
{
    parent[src] = dest;
    __threadfence();
}

// initializing parent of every component to itself at start.
__global__ void initParent(int *parent, int V)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V)
    {
        parent[i] = i;
    }
}

// initializing minVertex[i] to maxValues with weight containing 10^6 which is represented by 32 leftbits and edgeNo is stored as
// E indicating default one indicating we did not find any minEdge till now.
__global__ void initMinVertex(long long int *minVertex, int E, int V)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V)
    {
        minVertex[i] = ((long long int)1000000 << 32) | (unsigned int)E;
    }
}

// finding minedge from every component
__global__ void findMinEdge(int *src, int *dest, int *weight, int *type, int *parent, long long int *minVertex, int *isEdgeVisited, int E)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < E && isEdgeVisited[i] == 0)
    {
        int srcParent = findParent(parent, src[i]);
        int destParent = findParent(parent, dest[i]);
        int wt = weight[i] * type[i];

        if (srcParent == destParent)
        {
            isEdgeVisited[i] = 1;
        }

        if (srcParent != destParent)
        {
            long long int wtAndIdx = ((long long int)wt << 32) | (unsigned int)i;
            atomicMin(&minVertex[srcParent], wtAndIdx);
            atomicMin(&minVertex[destParent], wtAndIdx);
            __threadfence();
        }
    }
}

// merging components based on minedge from each component.
__global__ void mergeComponents(int *src, int *dest, int *weight, int *type, int *parent, long long int *minVertex, int *isEdgeVisited, int E, int *nComp, int V)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V)
    {
        // extracting edgeNo from minVertex[i] where in first 32 bits from left it stores weight of edge and next 32 distinct edgeNo
        int edgeNo = (int)(minVertex[i] & 0xFFFFFFFF);
        int srcParent = findParent(parent, src[edgeNo]);
        int destParent = findParent(parent, dest[edgeNo]);

        if (srcParent != destParent && parent[i] == i && edgeNo != E && isEdgeVisited[edgeNo] == 0)
        {
            int minWt = (int)(minVertex[i] >> 32);
            int edgeVisited;

            if (srcParent == i)
            {
                edgeVisited = atomicCAS(&isEdgeVisited[edgeNo], 0, 1);
                if (edgeVisited == 0)
                {
                    atomicSub((unsigned int *)nComp, 1);
                    merge(parent, srcParent, destParent);
                    atomicAdd(&mstWeight, minWt);
                }
            }
            else if (destParent == i)
            {
                edgeVisited = atomicCAS(&isEdgeVisited[edgeNo], 0, 1);
                if (edgeVisited == 0)
                {
                    atomicSub((unsigned int *)nComp, 1);
                    merge(parent, destParent, srcParent);
                    atomicAdd(&mstWeight, minWt);
                }
            }
            __threadfence();
        }
    }
}

__global__ void modMST()
{
    mstWeight = mstWeight % d_MOD; // updating in this kerenel to avoid datarace.
}

int main()
{
    int V;
    cin >> V;
    int E;
    cin >> E;

    int finalWeight = 0;

    // allocating memory for edges on cpu
    Edges.src = (int *)malloc(E * sizeof(int));
    Edges.dest = (int *)malloc(E * sizeof(int));
    Edges.weight = (int *)malloc(E * sizeof(int));
    Edges.type = (int *)malloc(E * sizeof(int));

    // parsing inputs
    int i = 0, type = 1;
    while (i < E)
    {
        string s;
        cin >> Edges.src[i] >> Edges.dest[i] >> Edges.weight[i];
        cin >> s;

        if (s == "normal")
            type = 1;
        else if (s == "green")
            type = 2;
        else if (s == "dept")
            type = 3;
        else if (s == "traffic")
            type = 5;

        Edges.type[i] = type;
        i++;
    }

    int hostComp = V;
    int *dParent, *dEdgeVisited;
    int *nComp;
    long long int *dMinVertex;

    // all memory allocations on gpu
    cudaMalloc(&d_Edges.src, E * sizeof(int));
    cudaMalloc(&d_Edges.dest, E * sizeof(int));
    cudaMalloc(&d_Edges.weight, E * sizeof(int));
    cudaMalloc(&d_Edges.type, E * sizeof(int));
    cudaMalloc(&dParent, V * sizeof(int));
    cudaMalloc(&dMinVertex, V * sizeof(long long int));
    cudaMalloc(&dEdgeVisited, E * sizeof(int));
    cudaMalloc(&nComp, sizeof(int));

    // cuda mem copys and set
    cudaMemset(dEdgeVisited, 0, E * sizeof(int));
    cudaMemcpy(d_Edges.src, Edges.src, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Edges.dest, Edges.dest, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Edges.weight, Edges.weight, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Edges.type, Edges.type, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(nComp, &hostComp, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mstWeight, &finalWeight, sizeof(int));

    int threadsPerBlock = 1024;
    int vertexBlocks = ceil((V / (double)threadsPerBlock));
    int edgeBlocks = ceil((E / (double)threadsPerBlock));

    // keeping all kernel launches inside timing blocks.
    auto start = std::chrono::high_resolution_clock::now();
    initParent<<<vertexBlocks, threadsPerBlock>>>(dParent, V);
    while (hostComp > 1)
    {
        initMinVertex<<<vertexBlocks, threadsPerBlock>>>(dMinVertex, E, V);
        findMinEdge<<<edgeBlocks, threadsPerBlock>>>(d_Edges.src, d_Edges.dest, d_Edges.weight, d_Edges.type, dParent, dMinVertex, dEdgeVisited, E);
        mergeComponents<<<vertexBlocks, threadsPerBlock>>>(d_Edges.src, d_Edges.dest, d_Edges.weight, d_Edges.type, dParent, dMinVertex, dEdgeVisited, E, nComp, V);
        modMST<<<1, 1>>>();
        cudaMemcpy(&hostComp, nComp, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();

    // copying mstweight from gpu to cpu
    cudaMemcpyFromSymbol(&finalWeight, mstWeight, sizeof(int));
    cudaDeviceSynchronize();

    std::chrono::duration<double> elapsed1 = end - start;

    cudaFree(dEdgeVisited);
    cudaFree(d_Edges.src);
    cudaFree(d_Edges.dest);
    cudaFree(d_Edges.weight);
    cudaFree(d_Edges.type);
    cudaFree(dMinVertex);
    cudaFree(dParent);
    cudaFree(nComp);

    printf("%d\n", finalWeight);
    // cout << elapsed1.count() << " s\n";
    return 0;
}

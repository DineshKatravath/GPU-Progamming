#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <random>
#include <climits>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

using namespace std;

#define MAX_PATH_LENGTH 100   // to prevent memory issues and loops
#define MAX_RANDOM_ATTEMPTS 5 // Number of random neighbor attempts before giving up
#define INF (LLONG_MAX / 4)

// CUDA error checking helper function
#define CHECK_CUDA_ERROR(call)                                    \
    do                                                            \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// used for both large and small test cases
struct Drop
{
    int shelter;
    int prime;
    int elderly;

    // Default constructor needed for Thrust
    __host__ __device__ Drop() : shelter(0), prime(0), elderly(0) {}

    __host__ __device__ Drop(int s, int p, int e) : shelter(s), prime(p), elderly(e) {}
};

// CSR format for graph representation in large test cases
struct CSRGraph
{
    int *offsets;
    int *edges;
    int *weights;
    int num_nodes;
    int num_edges;

    // Constructor
    CSRGraph(int n, int e) : num_nodes(n), num_edges(e)
    {
        cudaMalloc(&offsets, sizeof(int) * (n + 1));
        cudaMalloc(&edges, sizeof(int) * e);
        cudaMalloc(&weights, sizeof(int) * e);
    }

    // Destructor
    ~CSRGraph()
    {
        cudaFree(offsets);
        cudaFree(edges);
        cudaFree(weights);
    }
};

// kernels for large test cases

// hash function for random selection
__device__ unsigned int hash_function(unsigned int *seed)
{
    *seed = *seed * 1664525 + 1013904223; // Linear generator
    return *seed;
}

// Get a random neighbor from the CSR graph
__device__ int getRandomNeighbor(const CSRGraph *graph, int current_node, unsigned int *seed, bool *visited, int max_attempts)
{
    int start = graph->offsets[current_node];
    int end = graph->offsets[current_node + 1];

    if (start == end)
    {
        return -1; // No neighbors
    }

    // Try to find unvisited neighbors only limited attempts to pick
    for (int i = 0; i < max_attempts; i++)
    {
        // Generate random index within the neighbor range
        int index = start + (hash_function(seed) % (end - start));
        int neighbor = graph->edges[index];

        if (!visited[neighbor])
        {
            return neighbor;
        }
    }

    // If all attempts failed , pick any random neighnour
    int index = start + (hash_function(seed) % (end - start));
    return graph->edges[index];
}

// Evacuate people from populated cities
__global__ void evacuatePopulatedCities(const CSRGraph *graph, int num_populated_cities, int *populated_city_ids, int *city_populations,
                                        int *shelter_ids, int *shelter_capacities, int num_shelters, int *shelter_occupancy, int max_elderly_distance, int *path_sizes,
                                        int *paths, int *num_drops, Drop *drops, int max_path_length, unsigned int *random_seeds)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_populated_cities)
        return;

    int city_id = populated_city_ids[tid];
    int prime_age = city_populations[2 * tid];
    int elderly = city_populations[2 * tid + 1];

    // return if no one is present
    if (prime_age == 0 && elderly == 0)
        return;

    int path_idx = 0;
    int drop_idx = 0;

    // Add the starting city to the path
    paths[tid * max_path_length + path_idx++] = city_id;

    // array to track if node is visited or not
    bool visited[100000] = {false};
    visited[city_id] = true;

    int current_city = city_id;
    int total_distance = 0;
    int remaining_prime = prime_age;
    int remaining_elderly = elderly;
    unsigned int seed = random_seeds[tid];

    // run loop until there is no population to move or it covers MAX_PATH_LENGTH steps
    while ((remaining_prime > 0 || remaining_elderly > 0) && path_idx < max_path_length)
    {
        bool has_shelter = false;
        int shelter_idx = -1;

        for (int s = 0; s < num_shelters; s++)
        {
            if (shelter_ids[s] == current_city)
            {
                has_shelter = true;
                shelter_idx = s;
                break;
            }
        }

        // If we are at a shelter, try to drop people
        if (has_shelter)
        {
            int capacity = shelter_capacities[shelter_idx];
            int current_occupancy = atomicAdd(&shelter_occupancy[shelter_idx], 0);
            int available_capacity = max(0, capacity - current_occupancy);

            // Prioritize elderly as they can only travel limited distance
            int elderly_to_shelter = min(remaining_elderly, available_capacity);
            int prime_to_shelter = min(remaining_prime, available_capacity - elderly_to_shelter);

            if (elderly_to_shelter > 0 || prime_to_shelter > 0)
            {
                atomicAdd(&shelter_occupancy[shelter_idx], elderly_to_shelter + prime_to_shelter);
                drops[tid * 1000 + drop_idx++] = Drop(current_city, prime_to_shelter, elderly_to_shelter);

                // Update remaining population
                remaining_elderly -= elderly_to_shelter;
                remaining_prime -= prime_to_shelter;
            }
        }
        if (remaining_prime == 0 && remaining_elderly == 0)
            break;

        // visit random neighbour next
        int next_city = getRandomNeighbor(graph, current_city, &seed, visited, MAX_RANDOM_ATTEMPTS);

        // If we cant find a neighbour
        if (next_city == -1 || next_city == current_city)
        {
            drops[tid * 1000 + drop_idx++] = Drop(current_city, remaining_prime, remaining_elderly);
            remaining_prime = 0;
            remaining_elderly = 0;
            break;
        }

        int edge_weight = 0;
        int start = graph->offsets[current_city];
        int end = graph->offsets[current_city + 1];

        for (int e = start; e < end; e++)
        {
            if (graph->edges[e] == next_city)
            {
                edge_weight = graph->weights[e];
                break;
            }
        }
        total_distance += edge_weight;

        // Check if elderly can continue based on max distance
        if (remaining_elderly > 0 && total_distance > max_elderly_distance)
        {
            drops[tid * 1000 + drop_idx++] = Drop(current_city, 0, remaining_elderly);
            remaining_elderly = 0;
        }

        // Add next city to path and move to next city
        paths[tid * max_path_length + path_idx++] = next_city;
        visited[next_city] = true;
        current_city = next_city;

        // if we find loop drop everyone there
        bool in_cycle = false;
        for (int i = 0; i < path_idx - 1; i++)
        {
            if (paths[tid * max_path_length + i] == current_city)
            {
                in_cycle = true;
                break;
            }
        }

        if (in_cycle)
        {
            drops[tid * 1000 + drop_idx++] = Drop(current_city, remaining_prime, remaining_elderly);
            remaining_prime = 0;
            remaining_elderly = 0;
            break;
        }
    }

    // If we've reached max path length but still have people, drop them in current city
    if (remaining_prime > 0 || remaining_elderly > 0)
    {
        drops[tid * 1000 + drop_idx++] = Drop(current_city, remaining_prime, remaining_elderly);
    }

    path_sizes[tid] = path_idx;
    num_drops[tid] = drop_idx;

    // saving random seed
    random_seeds[tid] = seed;
}

// kernels for small functions

// used for bellman ford to find shortest dist
__global__ void initDistParent(long long *dist, int *parent, long long inf, long long total)
{
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total)
        return;
    dist[tid] = inf;
    parent[tid] = -1;
}

__global__ void relaxEdges(int E, int S, int *u, int *v, int *w, long long *dist, int *parent, bool *changed, int num_cities)
{
    int s = blockIdx.y; // shelterIdx
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E)
        return;

    int src = u[idx];
    int dst = v[idx];
    long long *dist_s = dist + (long long)s * num_cities;
    int *parent_s = parent + (long long)s * num_cities;

    long long dsrc = dist_s[src];
    if (dsrc == INF)
        return;

    long long newdist = dsrc + (long long)w[idx];
    long long old = atomicMin(&dist_s[dst], newdist);

    if (newdist < old)
    {
        changed[s] = true;
        parent_s[dst] = src;
    }
}

// Updated simulateEvacuation kernel with proper bounds checking
__global__ void simulateEvacuation(
    int num_populated_cities,
    long long *distances, int *parents,
    int *city_pop, int *shelter_ids, int *shelter_caps,
    int max_elderly_dist, int num_cities, int num_shelters,
    int *pop_city_ids,
    int *d_shelter_occupancy,
    int *path_sizes,
    int *paths,
    int *num_drops,
    Drop *drops,
    int max_path_length,
    int *initdist)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_populated_cities)
        return;

    // Load city data
    int city_id = pop_city_ids[tid];
    int prime_age = city_pop[2 * tid];
    int elderly = city_pop[2 * tid + 1];

    // If no population to evacuate, return early
    if (prime_age == 0 && elderly == 0)
        return;

    // Initialize path and drop tracking
    int path_idx = 0;
    int drop_idx = 0;
    int remaining_prime = prime_age;
    int remaining_elderly = elderly;

    // Safety check to prevent buffer overflow
    if (tid * max_path_length >= num_populated_cities * max_path_length)
    {
        return; // Out of bounds, exit
    }

    // Start with the source city in the path
    paths[tid * max_path_length + path_idx++] = city_id;

    // Keep trying shelters until all people are evacuated or no more options
    int current_city = city_id;
    int shelter_attempts = 0;
    const int MAX_SHELTER_ATTEMPTS = min(50, num_shelters); // Limit shelter attempts
    int prev_shelter_idx = -1;
    int best_shelter_idx = -1;

    while ((remaining_prime > 0 || remaining_elderly > 0) &&
           shelter_attempts < MAX_SHELTER_ATTEMPTS &&
           path_idx < max_path_length - 1) // Ensure room for one more city
    {
        // Find the shelter with the best capacity/distance ratio from current city
        prev_shelter_idx = best_shelter_idx;
        best_shelter_idx = -1;
        float best_ratio = -1.0;

        // Skip any shelters we've already tried
        for (int s = 0; s < num_shelters; s++)
        {
            // Bounds check for distance array
            long long idx = s * num_cities + current_city;
            if (idx < 0 || idx >= (long long)num_shelters * num_cities)
                continue;

            long long dist = distances[idx];
            if (dist != INF && s != prev_shelter_idx)
            {
                // Get current occupancy to consider available capacity
                int current_occupancy = atomicAdd(&d_shelter_occupancy[s], 0);
                int available_capacity = max(0, shelter_caps[s] - current_occupancy);

                // Calculate ratio: capacity / (distance + 1)
                float ratio = (float)available_capacity / (float)(dist + 1);

                if (available_capacity > 0 && ratio > best_ratio)
                {
                    best_ratio = ratio;
                    best_shelter_idx = s;
                }
            }
        }

        // If no reachable shelter with capacity, drop everyone at current city
        if (best_shelter_idx == -1 || best_ratio <= 0)
        {
            if (remaining_prime > 0 || remaining_elderly > 0)
            {
                // Bounds check for drops array
                if (tid * num_cities + drop_idx < num_populated_cities * num_cities)
                {
                    drops[tid * num_cities + drop_idx++] = Drop(current_city, remaining_prime, remaining_elderly);
                    remaining_prime = 0;
                    remaining_elderly = 0;
                }
            }
            break;
        }

        int shelter_city = shelter_ids[best_shelter_idx];

        // If current city is already the target shelter, try to accommodate people
        if (current_city == shelter_city)
        {
            int shelter_capacity = shelter_caps[best_shelter_idx];
            int current_occupancy = atomicAdd(&d_shelter_occupancy[best_shelter_idx], 0);
            int available_capacity = max(0, shelter_capacity - current_occupancy);

            // Prioritize elderly for shelter space
            int elderly_to_shelter = min(remaining_elderly, available_capacity);
            int prime_to_shelter = min(remaining_prime, available_capacity - elderly_to_shelter);

            // Update shelter occupancy
            if (elderly_to_shelter > 0 || prime_to_shelter > 0)
            {
                atomicAdd(&d_shelter_occupancy[best_shelter_idx], elderly_to_shelter + prime_to_shelter);

                // Bounds check for drops array
                if (tid * num_cities + drop_idx < num_populated_cities * num_cities)
                {
                    drops[tid * num_cities + drop_idx++] = Drop(current_city, prime_to_shelter, elderly_to_shelter);
                    remaining_elderly -= elderly_to_shelter;
                    remaining_prime -= prime_to_shelter;
                }
            }

            // Look for next shelter in next iteration
            shelter_attempts++;
            continue;
        }

        // Simulate evacuation along the path to this shelter
        bool reached_shelter = false;

        // Local variables to track changes during path traversal
        int local_remaining_prime = remaining_prime;
        int local_remaining_elderly = remaining_elderly;
        int local_drop_idx = drop_idx;
        int local_path_idx = path_idx;

        // Build path to shelter using parent pointers
        int current = current_city;
        long long path_dist = 0;

        // Path finding and population management while traveling to shelter
        while (current != shelter_city)
        {
            // Check if any city along the way is a shelter and drop evacuees if possible
            for (int s = 0; s < num_shelters; s++)
            {
                if (shelter_ids[s] == current)
                {
                    int shelter_capacity = shelter_caps[s];
                    int current_occupancy = atomicAdd(&d_shelter_occupancy[s], 0);
                    int available_capacity = max(0, shelter_capacity - current_occupancy);

                    // Prioritize elderly for intermediate shelter space
                    int elderly_to_shelter = min(local_remaining_elderly, available_capacity);
                    int prime_to_shelter = min(local_remaining_prime, available_capacity - elderly_to_shelter);

                    if (elderly_to_shelter > 0 || prime_to_shelter > 0)
                    {
                        atomicAdd(&d_shelter_occupancy[s], elderly_to_shelter + prime_to_shelter);

                        // Bounds check for drops array
                        if (tid * num_cities + local_drop_idx < num_populated_cities * num_cities)
                        {
                            drops[tid * num_cities + local_drop_idx++] = Drop(current, prime_to_shelter, elderly_to_shelter);
                            local_remaining_elderly -= elderly_to_shelter;
                            local_remaining_prime -= prime_to_shelter;
                        }
                    }
                }
            }

            // Find next city in path using parent pointers
            long long parent_idx = best_shelter_idx * num_cities + current;
            if (parent_idx < 0 || parent_idx >= (long long)num_shelters * num_cities)
            {
                break; // Out of bounds, exit loop
            }

            int next = parents[parent_idx];

            // If no path exists or reached max path length, break
            if (next == -1 || local_path_idx >= max_path_length - 1)
            {
                break;
            }

            // Calculate distance to next city - safe index check
            long long edge_idx = current * num_cities + next;
            if (edge_idx < 0 || edge_idx >= num_cities * num_cities)
            {
                break; // Out of bounds, exit loop
            }

            long long edge_dist = initdist[edge_idx];
            path_dist += edge_dist;

            // Calculate distance to next shelter - safe index check
            long long shelter_dist_idx = best_shelter_idx * num_cities + current_city;
            if (shelter_dist_idx < 0 || shelter_dist_idx >= (long long)num_shelters * num_cities)
            {
                break; // Out of bounds, exit loop
            }

            long long next_shelter_dist = distances[shelter_dist_idx];

            // Check if elderly can continue based on max distance
            if (local_remaining_elderly > 0 && current != current_city && next_shelter_dist > max_elderly_dist)
            {
                // Drop elderly at current city - safe index check
                if (tid * num_cities + local_drop_idx < num_populated_cities * num_cities)
                {
                    drops[tid * num_cities + local_drop_idx++] = Drop(current, 0, local_remaining_elderly);
                    local_remaining_elderly = 0;
                }
            }

            if (local_remaining_elderly > 0 && current == current_city && path_dist > max_elderly_dist)
            {
                // Drop elderly at current city - safe index check
                if (tid * num_cities + local_drop_idx < num_populated_cities * num_cities)
                {
                    atomicAdd(&d_shelter_occupancy[best_shelter_idx], local_remaining_elderly);
                    drops[tid * num_cities + local_drop_idx++] = Drop(current, 0, local_remaining_elderly);
                    local_remaining_elderly = 0;
                }
            }

            if (local_remaining_elderly <= 0 && local_remaining_prime <= 0)
            {
                break;
            }

            // Add next city to path - safe index check
            if (tid * max_path_length + local_path_idx < num_populated_cities * max_path_length)
            {
                paths[tid * max_path_length + local_path_idx++] = next;
            }

            // Update current city
            current = next;

            // If reached target shelter
            if (current == shelter_city)
            {
                reached_shelter = true;
                break;
            }
        }

        // If shelter was reached, try to accommodate people
        if (reached_shelter)
        {
            int shelter_capacity = shelter_caps[best_shelter_idx];
            int current_occupancy = atomicAdd(&d_shelter_occupancy[best_shelter_idx], 0);
            int available_capacity = max(0, shelter_capacity - current_occupancy);

            // Prioritize elderly for shelter space
            int elderly_to_shelter = min(local_remaining_elderly, available_capacity);
            int prime_to_shelter = min(local_remaining_prime, available_capacity - elderly_to_shelter);

            // Update shelter occupancy and record drop
            if (elderly_to_shelter > 0 || prime_to_shelter > 0)
            {
                atomicAdd(&d_shelter_occupancy[best_shelter_idx], elderly_to_shelter + prime_to_shelter);

                // Bounds check for drops array
                if (tid * num_cities + local_drop_idx < num_populated_cities * num_cities)
                {
                    drops[tid * num_cities + local_drop_idx++] = Drop(shelter_city, prime_to_shelter, elderly_to_shelter);
                    local_remaining_elderly -= elderly_to_shelter;
                    local_remaining_prime -= prime_to_shelter;
                }
            }

            // Update main tracking variables
            remaining_elderly = local_remaining_elderly;
            remaining_prime = local_remaining_prime;
            drop_idx = local_drop_idx;
            path_idx = local_path_idx;
            max_elderly_dist -= path_dist;
            current_city = shelter_city;
        }
        else
        {
            // Update main tracking variables even if shelter wasn't reached
            remaining_elderly = local_remaining_elderly;
            remaining_prime = local_remaining_prime;
            drop_idx = local_drop_idx;
            path_idx = local_path_idx;
            max_elderly_dist -= path_dist;
            break;
        }

        shelter_attempts++;
    }

    // If there are still people left, drop them at the last city reached
    if (remaining_prime > 0 || remaining_elderly > 0)
    {
        // Bounds check for drops array
        if (tid * num_cities + drop_idx < num_populated_cities * num_cities)
        {
            drops[tid * num_cities + drop_idx++] = Drop(current_city, remaining_prime, remaining_elderly);
        }
    }

    // Update final path size and drop count with bounds checking
    if (tid < num_populated_cities)
    {
        path_sizes[tid] = min(path_idx, max_path_length);
        num_drops[tid] = min(drop_idx, num_cities);
    }
}

// This function creates a new vector with only valid path elements
thrust::host_vector<int> extract_valid_path(const thrust::host_vector<int> &paths, int start_idx, int path_len, int max_path_len)
{
    thrust::host_vector<int> result;
    for (int i = 0; i < path_len; i++)
    {
        if (start_idx * max_path_len + i >= paths.size())
        {
            break;
        }
        int val = paths[start_idx * max_path_len + i];
        if (val >= 0)
        {
            result.push_back(val);
        }
    }
    return result;
}

// This function creates a new vector with only valid drops
thrust::host_vector<Drop> extract_valid_drops(const thrust::host_vector<Drop> &drops, int start_idx, int drop_count, int num_cities)
{
    thrust::host_vector<Drop> result;
    for (int i = 0; i < drop_count; i++)
    {
        if (start_idx * num_cities + i >= drops.size())
        {
            break;
        }
        const Drop &drop = drops[start_idx * num_cities + i];
        if (drop.shelter != 0 || drop.prime != 0 || drop.elderly != 0)
        {
            result.push_back(drop);
        }
    }
    return result;
}

// cpu side code

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    ifstream infile(argv[1]);
    if (!infile)
    {
        cerr << "Error: Cannot open file " << argv[1] << "\n";
        return 1;
    }

    // Read input file
    int num_cities;
    infile >> num_cities;

    int num_roads;
    infile >> num_roads;

    // Store roads as a flat array: [u1, v1, length1, capacity1, u2, v2, length2, capacity2, ...]
    int *roads = new int[num_roads * 4];

    cout << "Input file contains " << num_cities << " cities and " << num_roads << " roads" << endl;

    // Create CSR format on host for large testcases
    vector<vector<pair<int, int>>> adjacency_list(num_cities);

    for (int i = 0; i < num_roads; i++)
    {
        infile >> roads[4 * i] >> roads[4 * i + 1] >> roads[4 * i + 2] >> roads[4 * i + 3];

        // Add edges in both directions (undirected graph)
        adjacency_list[roads[4 * i]].push_back({roads[4 * i + 1], roads[4 * i + 2]});
        adjacency_list[roads[4 * i + 1]].push_back({roads[4 * i], roads[4 * i + 2]});
    }

    // Create CSR arrays on host
    vector<int> h_offsets(num_cities + 1);
    vector<int> h_edges;
    vector<int> h_weights;

    h_offsets[0] = 0;
    for (int i = 0; i < num_cities; i++)
    {
        h_offsets[i + 1] = h_offsets[i] + adjacency_list[i].size();
        for (auto &edge : adjacency_list[i])
        {
            h_edges.push_back(edge.first);
            h_weights.push_back(edge.second);
        }
    }

    int total_edges = h_edges.size();
    cout << "Total edges in CSR: " << total_edges << endl;

    int num_shelters;
    infile >> num_shelters;

    int *h_shelter_city = new int[num_shelters];
    int *h_shelter_capacity = new int[num_shelters];

    for (int i = 0; i < num_shelters; i++)
    {
        infile >> h_shelter_city[i] >> h_shelter_capacity[i];
    }

    int num_populated_cities;
    infile >> num_populated_cities;

    // Store populated cities separately
    int *h_pop_city_ids = new int[num_populated_cities];
    int *h_city_pop = new int[num_populated_cities * 2]; // Flattened [prime-age, elderly] pairs

    for (int i = 0; i < num_populated_cities; i++)
    {
        infile >> h_pop_city_ids[i] >> h_city_pop[2 * i] >> h_city_pop[2 * i + 1];
    }

    int max_distance_elderly;
    infile >> max_distance_elderly;

    infile.close();

    int maxPathLen = min(5 * (int)num_cities, 1000); // Cap at 1000 to prevent excessive memory usage

    cout << "Input reading complete" << endl;
    cout << "Number of cities: " << num_cities << endl;
    cout << "Number of roads: " << num_roads << endl;
    cout << "Number of shelters: " << num_shelters << endl;
    cout << "Number of populated cities: " << num_populated_cities << endl;
    cout << "Max elderly distance: " << max_distance_elderly << endl;

    thrust::host_vector<int> h_path_sizes;
    thrust::host_vector<int> h_paths;
    thrust::host_vector<int> h_num_drops;
    thrust::host_vector<Drop> h_drops;
    thrust::host_vector<int> h_final_shelter_occupancy;

    if (num_cities > 20)
    {
        maxPathLen = MAX_PATH_LENGTH;
        // Create CSR graph on device
        CSRGraph *d_graph;
        CHECK_CUDA_ERROR(cudaMalloc(&d_graph, sizeof(CSRGraph)));
        CSRGraph h_graph(num_cities, total_edges);

        // Copy CSR data to device
        CHECK_CUDA_ERROR(cudaMemcpy(h_graph.offsets, h_offsets.data(), sizeof(int) * (num_cities + 1), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(h_graph.edges, h_edges.data(), sizeof(int) * total_edges, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(h_graph.weights, h_weights.data(), sizeof(int) * total_edges, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_graph, &h_graph, sizeof(CSRGraph), cudaMemcpyHostToDevice));

        // Copy city and shelter data to device

        thrust::device_vector<int> d_populated_city_ids(num_populated_cities);
        thrust::device_vector<int> d_populations(num_populated_cities);
        thrust::device_vector<int> d_shelter_ids(num_shelters);
        thrust::device_vector<int> d_shelter_capacities(num_shelters);

        // Copy data from host raw pointers to device vectors
        cudaMemcpy(thrust::raw_pointer_cast(d_populated_city_ids.data()), h_pop_city_ids, sizeof(int) * num_populated_cities, cudaMemcpyHostToDevice);
        cudaMemcpy(thrust::raw_pointer_cast(d_populations.data()), h_city_pop, sizeof(int) * num_populated_cities, cudaMemcpyHostToDevice);
        cudaMemcpy(thrust::raw_pointer_cast(d_shelter_ids.data()), h_shelter_city, sizeof(int) * num_shelters, cudaMemcpyHostToDevice);
        cudaMemcpy(thrust::raw_pointer_cast(d_shelter_capacities.data()), h_shelter_capacity, sizeof(int) * num_shelters, cudaMemcpyHostToDevice);
        thrust::device_vector<int> d_shelter_occupancy(num_shelters, 0);

        // Initialize random seeds for each thread
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<unsigned int> dist(1, UINT_MAX);

        thrust::host_vector<unsigned int> h_random_seeds(num_populated_cities);
        for (int i = 0; i < num_populated_cities; i++)
        {
            h_random_seeds[i] = dist(gen);
        }
        thrust::device_vector<unsigned int> d_random_seeds = h_random_seeds;

        // Allocate memory for results
        thrust::device_vector<int> d_path_sizes(num_populated_cities, 0);
        thrust::device_vector<int> d_paths(num_populated_cities * MAX_PATH_LENGTH, -1);
        thrust::device_vector<int> d_num_drops(num_populated_cities, 0);
        thrust::device_vector<Drop> d_drops(num_populated_cities * 1000); // Allow up to 1000 drops per city

        cout << "Starting evacuation simulation..." << endl;

        // Launch evacuation kernel
        int threads_per_block = 256;
        int blocks = (num_populated_cities + threads_per_block - 1) / threads_per_block;

        evacuatePopulatedCities<<<blocks, threads_per_block>>>(d_graph, num_populated_cities, thrust::raw_pointer_cast(d_populated_city_ids.data()),
                                                               thrust::raw_pointer_cast(d_populations.data()), thrust::raw_pointer_cast(d_shelter_ids.data()), thrust::raw_pointer_cast(d_shelter_capacities.data()),
                                                               num_shelters, thrust::raw_pointer_cast(d_shelter_occupancy.data()), max_distance_elderly, thrust::raw_pointer_cast(d_path_sizes.data()),
                                                               thrust::raw_pointer_cast(d_paths.data()), thrust::raw_pointer_cast(d_num_drops.data()), thrust::raw_pointer_cast(d_drops.data()),
                                                               MAX_PATH_LENGTH, thrust::raw_pointer_cast(d_random_seeds.data()));

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        cout << "Evacuation simulation complete, copying results back to host" << endl;

        // Copy results back to host
        h_path_sizes = d_path_sizes;
        h_paths = d_paths;
        h_num_drops = d_num_drops;
        h_drops = d_drops;
        h_final_shelter_occupancy = d_shelter_occupancy;

        // // Calculate statistics
        // long long totalPrime = 0, totalElderly = 0;
        // long long totalPrimeSaved = 0, totalElderlySaved = 0;

        // for (int i = 0; i < num_populated_cities; i++)
        // {
        //     totalPrime += h_city_pop[2 * i];
        //     totalElderly += h_city_pop[2 * i + 1];
        // }

        // thrust::host_vector<long long> savedShelter(num_shelters, 0);

        // // Calculate statistics using our results
        // for (int i = 0; i < num_populated_cities; i++)
        // {
        //     for (int j = 0; j < h_num_drops[i]; j++)
        //     {
        //         const Drop &drop = h_drops[i * 1000 + j];
        //         for (int s = 0; s < num_shelters; s++)
        //         {
        //             if (h_shelter_city[s] == drop.shelter)
        //             {
        //                 totalPrimeSaved += drop.prime;
        //                 totalElderlySaved += drop.elderly;
        //                 savedShelter[s] += drop.prime + drop.elderly;
        //                 break;
        //             }
        //         }
        //     }
        // }

        // // Calculate total saved and penalties
        // long long totalSaved = 0, totalPenalty = 0;
        // for (int j = 0; j < num_shelters; j++)
        // {
        //     long long capacity = static_cast<long long>(h_shelter_capacity[j]);
        //     long long saved = savedShelter[j];
        //     long long overflow = std::max(0LL, saved - capacity);
        //     long long penalty = std::min(overflow, capacity);
        //     totalSaved += std::max(0LL, saved - penalty);
        //     totalPenalty += penalty;
        // }

        // // Print statistics
        // printf("Total prime age: %lld, saved: %lld (%.2f%%)\n",
        //     totalPrime, totalPrimeSaved,
        //     totalPrime > 0 ? (100.0 * totalPrimeSaved / totalPrime) : 0.0);

        // printf("Total elderly: %lld, saved: %lld (%.2f%%)\n",
        //     totalElderly, totalElderlySaved,
        //     totalElderly > 0 ? (100.0 * totalElderlySaved / totalElderly) : 0.0);

        // printf("Total saved: %lld, Total penalty: %lld\n\n", totalSaved, totalPenalty);

        // Write results to output file
        std::ofstream outfile(argv[2]);
        if (!outfile)
        {
            std::cerr << "Error: Cannot open output file " << argv[2] << "\n";
            return 1;
        }

        // Output paths per city
        for (long long i = 0; i < num_populated_cities; i++)
        {
            long long currentPathSize = h_path_sizes[i];
            for (long long j = 0; j < currentPathSize; j++)
            {
                outfile << h_paths[i * MAX_PATH_LENGTH + j] << " ";
            }
            outfile << "\n";
        }

        // Output drops per city
        for (long long i = 0; i < num_populated_cities; i++)
        {
            long long currentDropSize = h_num_drops[i];
            for (long long j = 0; j < currentDropSize; j++)
            {
                const Drop &drop = h_drops[i * 1000 + j];
                outfile << drop.shelter << " " << drop.prime << " " << drop.elderly << " ";
            }
            outfile << "\n";
        }
        outfile.close();
        cout << "Results written to " << argv[2] << endl;
    }
    else
    {
        const int N = num_cities;
        int *initDistances = new int[N * N];
        std::fill(initDistances, initDistances + N * N, INF);

        // Populate from edge list first
        for (int i = 0; i < num_roads; ++i)
        {
            int src = roads[4 * i];
            int dst = roads[4 * i + 1];
            initDistances[src * N + dst] = roads[4 * i + 2];
            initDistances[dst * N + src] = roads[4 * i + 2];
        }

        // Allocate and copy to device memory
        int *d_initDistances;
        CHECK_CUDA_ERROR(cudaMalloc(&d_initDistances, sizeof(int) * N * N));
        CHECK_CUDA_ERROR(cudaMemcpy(d_initDistances, initDistances, sizeof(int) * N * N, cudaMemcpyHostToDevice));

        // Building directed edges as we are running bellmanford
        int E = num_roads * 2;
        thrust::host_vector<int> h_u(E);
        thrust::host_vector<int> h_v(E);
        thrust::host_vector<int> h_w(E);

        for (int i = 0; i < num_roads; ++i)
        {
            int u = roads[4 * i], v = roads[4 * i + 1], w = roads[4 * i + 2];

            h_u[2 * i] = u;
            h_v[2 * i] = v;
            h_w[2 * i] = w;
            h_u[2 * i + 1] = v;
            h_v[2 * i + 1] = u;
            h_w[2 * i + 1] = w;
        }

        // Move data to device using Thrust
        thrust::device_vector<int> d_u = h_u;
        thrust::device_vector<int> d_v = h_v;
        thrust::device_vector<int> d_w = h_w;

        // Allocate device memory for distances and parents
        long long totalPairs = (long long)num_shelters * num_cities;
        thrust::device_vector<long long> d_dist(totalPairs);
        thrust::device_vector<int> d_parent(totalPairs);
        thrust::device_vector<bool> d_changed(num_shelters);

        // Initialize dist/parent
        int initThreads = 1024;
        int initBlocks = (totalPairs + initThreads - 1) / initThreads;
        initDistParent<<<initBlocks, initThreads>>>(thrust::raw_pointer_cast(d_dist.data()), thrust::raw_pointer_cast(d_parent.data()),
                                                    (long long)INF, totalPairs);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Initialize each shelter's own city distance to zero on device
        for (int s = 0; s < num_shelters; ++s)
        {
            long long sc = h_shelter_city[s];
            d_dist[(long long)s * num_cities + sc] = 0;
        }

        // Reset changed flags to false
        thrust::fill(d_changed.begin(), d_changed.end(), false);

        // Parallel Bellman-Ford for all shelters
        int threads = 1024;
        int blocksE = (E + threads - 1) / threads;
        thrust::host_vector<bool> h_changed(num_shelters);
        dim3 grid(blocksE, num_shelters);

        cout << "Running Bellman-Ford..." << endl;

        for (int iter = 0; iter < num_cities - 1; ++iter)
        {
            thrust::fill(d_changed.begin(), d_changed.end(), false);

            relaxEdges<<<grid, threads>>>(E, num_shelters, thrust::raw_pointer_cast(d_u.data()), thrust::raw_pointer_cast(d_v.data()),
                                          thrust::raw_pointer_cast(d_w.data()), thrust::raw_pointer_cast(d_dist.data()), thrust::raw_pointer_cast(d_parent.data()),
                                          thrust::raw_pointer_cast(d_changed.data()), num_cities);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            h_changed = d_changed;
            bool anyChanged = false;
            for (int i = 0; i < num_shelters; ++i)
            {
                if (h_changed[i])
                {
                    anyChanged = true;
                    break;
                }
            }

            if (!anyChanged)
            {
                cout << "Bellman-Ford converged after " << iter + 1 << " iterations" << endl;
                break;
            }
        }

        // Get results back from device
        thrust::host_vector<long long> h_dist = d_dist;
        thrust::host_vector<int> h_parent = d_parent;

        // Copy population & shelter info to device
        thrust::device_vector<int> d_city_pop(h_city_pop, h_city_pop + 2 * num_populated_cities);
        thrust::device_vector<int> d_shelter_ids(h_shelter_city, h_shelter_city + num_shelters);
        thrust::device_vector<int> d_shelter_caps(h_shelter_capacity, h_shelter_capacity + num_shelters);
        thrust::device_vector<int> d_pop_city_ids(h_pop_city_ids, h_pop_city_ids + num_populated_cities);

        // Initialize shelter occupancy
        thrust::device_vector<int> d_shelter_occupancy(num_shelters, 0);

        thrust::device_vector<int> d_path_sizes(num_populated_cities, 0);
        thrust::device_vector<int> d_paths(num_populated_cities * maxPathLen, -1);
        thrust::device_vector<int> d_num_drops(num_populated_cities, 0);
        thrust::device_vector<Drop> d_drops(num_populated_cities * num_cities);

        cout << "Running evacuation simulation..." << endl;

        // Launch evacuation
        int evacThreads = 256;
        int evacB = (num_populated_cities + evacThreads - 1) / evacThreads;
        simulateEvacuation<<<evacB, evacThreads>>>(num_populated_cities, thrust::raw_pointer_cast(d_dist.data()), thrust::raw_pointer_cast(d_parent.data()),
                                                   thrust::raw_pointer_cast(d_city_pop.data()), thrust::raw_pointer_cast(d_shelter_ids.data()), thrust::raw_pointer_cast(d_shelter_caps.data()),
                                                   max_distance_elderly, num_cities, num_shelters, thrust::raw_pointer_cast(d_pop_city_ids.data()), thrust::raw_pointer_cast(d_shelter_occupancy.data()),
                                                   thrust::raw_pointer_cast(d_path_sizes.data()), thrust::raw_pointer_cast(d_paths.data()), thrust::raw_pointer_cast(d_num_drops.data()),
                                                   thrust::raw_pointer_cast(d_drops.data()), maxPathLen, d_initDistances);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        cout << "Evacuation simulation completed" << endl;

        // Copy back results using Thrust's built-in data transfers
        h_path_sizes = d_path_sizes;
        h_paths = d_paths;
        h_num_drops = d_num_drops;
        h_drops = d_drops;

        // Print some diagnostics
        cout << "Path sizes: ";
        for (int i = 0; i < min(5, num_populated_cities); i++)
        {
            cout << h_path_sizes[i] << " ";
        }
        cout << "..." << endl;

        cout << "Drop counts: ";
        for (int i = 0; i < min(5, num_populated_cities); i++)
        {
            cout << h_num_drops[i] << " ";
        }
        cout << "..." << endl;

        // Now create dynamic vectors using Thrust's push_back
        std::vector<thrust::host_vector<int>> path_vectors;
        std::vector<thrust::host_vector<Drop>> drop_vectors;

        // Process each city's results
        for (int i = 0; i < num_populated_cities; i++)
        {
            // Create dynamic vectors with push_back for paths
            thrust::host_vector<int> city_path = extract_valid_path(h_paths, i, h_path_sizes[i], maxPathLen);
            path_vectors.push_back(city_path);

            // Create dynamic vectors with push_back for drops
            thrust::host_vector<Drop> city_drops = extract_valid_drops(h_drops, i, h_num_drops[i], num_cities);
            drop_vectors.push_back(city_drops);
        }

        // Process and print results
        // long long totalPrime = 0, totalElderly = 0, totalPrimeSaved = 0, totalElderlySaved = 0, totalSaved = 0, totalPenalty = 0;

        // for (int i = 0; i < num_populated_cities; i++)
        // {
        //     totalPrime += h_city_pop[2 * i];
        //     totalElderly += h_city_pop[2 * i + 1];
        // }

        // thrust::host_vector<long long> savedShelter(num_shelters, 0);

        // // Calculate statistics using our dynamic vectors
        // for (int i = 0; i < num_populated_cities; i++)
        // {
        //     for (size_t j = 0; j < drop_vectors[i].size(); j++)
        //     {
        //         const Drop &drop = drop_vectors[i][j];
        //         for (int s = 0; s < num_shelters; s++)
        //         {
        //             if (h_shelter_city[s] == drop.shelter)
        //             {
        //                 totalPrimeSaved += drop.prime;
        //                 totalElderlySaved += drop.elderly;
        //                 savedShelter[s] += drop.prime + drop.elderly;
        //                 break;
        //             }
        //         }
        //     }
        // }

        // for (int j = 0; j < num_shelters; j++)
        // {
        //     long long capacity = static_cast<long long>(h_shelter_capacity[j]);
        //     long long saved = savedShelter[j];
        //     long long overflow = std::max(0LL, saved - capacity);
        //     long long penalty = std::min(overflow, capacity); // If that logic is required
        //     totalSaved += std::max(0LL, saved - penalty);
        //     totalPenalty += penalty;
        // }

        // printf("total_prime : %lld , saved %lld \n", totalPrime, totalPrimeSaved);
        // printf("total_elder : %lld , saved %lld \n", totalElderly, totalElderlySaved);
        // printf("totalSave : %lld , totalPenalty %lld\n\n", totalSaved, totalPenalty);

        // // printf("total_pop_cities: %d\n", num_populated_cities);
        // // for (int i = 0; i < num_populated_cities; i++)
        // // {
        // //     printf("city_%d_path_len: %d\n", i, h_path_sizes[i]);

        // //     printf("city_%d_path: [", i);
        // //     for (size_t j = 0; j < path_vectors[i].size(); j++)
        // //     {
        // //         printf("%d", path_vectors[i][j]);
        // //         if (j + 1 < path_vectors[i].size())
        // //             printf(" ");
        // //     }
        // //     printf("]\n");

        // //     printf("city_%d_drops_count: %zu\n", i, drop_vectors[i].size());

        // //     printf("city_%d_drops: [\n", i);
        // //     for (size_t j = 0; j < drop_vectors[i].size(); j++)
        // //     {
        // //         const Drop &drop = drop_vectors[i][j];
        // //         printf("  {shelter: %d, prime: %d, elderly: %d}%s\n",
        // //             drop.shelter, drop.prime, drop.elderly,
        // //             j + 1 == drop_vectors[i].size() ? "" : ",");
        // //     }
        // //     printf("]\n");
        // // }

        std::ofstream outfile(argv[2]);
        if (!outfile)
        {
            std::cerr << "Error: Cannot open output file " << argv[2] << "\n";
            return 1;
        }

        // Output paths per city
        for (long long i = 0; i < num_populated_cities; i++)
        {
            long long currentPathSize = h_path_sizes[i];
            for (long long j = 0; j < currentPathSize; j++)
            {
                outfile << path_vectors[i][j] << " ";
            }
            outfile << "\n";
        }

        // Output drops per city
        for (long long i = 0; i < num_populated_cities; i++)
        {
            long long currentDropSize = drop_vectors[i].size();
            for (long long j = 0; j < currentDropSize; j++)
            {
                const Drop &drop = drop_vectors[i][j];
                outfile << drop.shelter << " " << drop.prime << " " << drop.elderly << " ";
            }
            outfile << "\n";
        }
        cout << "Results written to " << argv[2] << endl;

        // Clean up host memory
        delete[] roads;
        delete[] h_shelter_city;
        delete[] h_shelter_capacity;
        delete[] h_pop_city_ids;
        delete[] h_city_pop;
        delete[] initDistances;
        cudaFree(d_initDistances);
    }
    return 0;
}

#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <atomic>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{
    std::atomic<int> new_count_atomic{0};

    int new_distance = distances[frontier->vertices[0]] + 1;

    #pragma omp parallel for schedule(dynamic, 100)
    for (int i=0; i<frontier->count; i++) {
        
        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];
            
            if (distances[outgoing] != NOT_VISITED_MARKER) {
                continue;
            }

            if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, new_distance)) {
                // int index;
                // index = new_frontier->count;
                // while(!__sync_bool_compare_and_swap(&new_frontier->count, index, index + 1)){
                //     index = new_frontier->count;
                // }
                int index = new_count_atomic.fetch_add(1);
                new_frontier->vertices[index] = outgoing;
            }
        }
    }
    new_frontier->count = new_count_atomic;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}


int bottom_up_step(Graph g, bool* visit_mem, bool* new_visit_mem, int* distances) {
    std::atomic<int> new_frontier_size_atomic{0};
    // int new_frontier_size = 0;

    #pragma omp parallel for schedule(dynamic, 800) 
    for (int i=0; i<g->num_nodes; i++) {
        
        if (distances[i] != NOT_VISITED_MARKER) continue;

        int node_begin = g->incoming_starts[i];
        int node_end_ind = (i == g->num_nodes - 1)
                           ? g->num_edges
                           : g->incoming_starts[i + 1];
        int node_end = node_end_ind;
            
        for (int neighbor=node_begin; neighbor<node_end; neighbor++) {
            int neighbor_ind = g->incoming_edges[neighbor];
            if (visit_mem[neighbor_ind]) {
                // printf("FIND!\n");
                new_visit_mem[i] = true;
                distances[i] = distances[neighbor_ind] + 1;
                new_frontier_size_atomic.fetch_add(1);
                // new_frontier_size += 1;
                break;
            }

        }
    }

    return new_frontier_size_atomic;
    // return new_frontier_size;
}

void bfs_bottom_up(Graph graph, solution* sol) {

    bool* visit_mem = (bool*)malloc(sizeof(bool) * graph->num_nodes);
    bool* new_visit_mem = (bool*)malloc(sizeof(bool) * graph->num_nodes);

    memset(visit_mem, false, sizeof(bool) * graph->num_nodes);
    memset(new_visit_mem, false, sizeof(bool) * graph->num_nodes);
    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    // setup frontier with the root node
    visit_mem[ROOT_NODE_ID] = true;
    sol->distances[ROOT_NODE_ID] = 0;

    int frontier_size = 1;
    while (frontier_size != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        frontier_size = bottom_up_step(graph, visit_mem, new_visit_mem, sol->distances);

        // printf("frontier_size=%d\n", frontier_size);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
        
        bool* tmp;
        tmp = visit_mem;
        visit_mem = new_visit_mem;
        new_visit_mem = tmp;
        memset(new_visit_mem, false, sizeof(bool) * graph->num_nodes);

        // for (int i = 0; i < graph->num_nodes; i++) {
        //     visit_mem[i] = new_visit_mem[i];
        //     new_visit_mem[i] = false;
        // }
    }
}


void bfs_hybrid(Graph graph, solution* sol) {
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;
    vertex_set_clear(new_frontier);
    bool* visit_mem = (bool*)malloc(sizeof(bool) * graph->num_nodes);
    bool* new_visit_mem = (bool*)malloc(sizeof(bool) * graph->num_nodes);

    memset(visit_mem, false, sizeof(bool) * graph->num_nodes);
    memset(new_visit_mem, false, sizeof(bool) * graph->num_nodes);

    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    // visit_mem[ROOT_NODE_ID] = true;

    bool using_top_down = true;

    int frontier_size = frontier->count;
    int average_edge = graph->num_edges / graph->num_nodes;
    int remaining_node = graph->num_nodes - 1;

    while (frontier_size != 0) {
        
        if (using_top_down) {
            top_down_step(graph, frontier, new_frontier, sol->distances);
            vertex_set* tmp = frontier;
            frontier = new_frontier;
            new_frontier = tmp;
            frontier_size = frontier->count;
            remaining_node -= frontier->count;
            vertex_set_clear(new_frontier);

            if (frontier_size > remaining_node * 0.3) {
                // printf("SWITCH at frontier_size = %d, remaining_node = %d \n", frontier_size, remaining_node);
                using_top_down = false;
                for (int i = 0; i < frontier_size; i++) {
                    visit_mem[frontier->vertices[i]] = true;
                }
            }
        }
        else {
            frontier_size = bottom_up_step(graph, visit_mem, new_visit_mem, sol->distances);
            bool* tmp;
            tmp = visit_mem;
            visit_mem = new_visit_mem;
            new_visit_mem = tmp;
            memset(new_visit_mem, false, sizeof(bool) * graph->num_nodes);
            if (frontier_size < remaining_node * 0.3) {
                // printf("SWITCH at frontier_size = %d, remaining_node = %d \n", frontier_size, remaining_node);
                using_top_down = true;
                vertex_set_clear(frontier);

                std::atomic<int> count_tmp{0};
                #pragma omp parallel for schedule(dynamic, 1000)
                for (int i = 0; i < graph->num_nodes; i++) {
                    if (visit_mem[i]) {
                        frontier->vertices[count_tmp.fetch_add(1)] = i;
                    }
                }
                frontier->count = count_tmp;
            }
        }


        
    }
}


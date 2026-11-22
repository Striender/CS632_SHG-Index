# Skip-Aware Cluster Prefetching for SHG

## 1. Overview

This proposal aims to improve the performance of the Shortcut-enabled
Hierarchical Graph (SHG) index for approximate nearest neighbor (ANN)
search by making memory prefetching aware of SHG's layer-skipping
decisions.

SHG uses hierarchical vector compression and learned shortcuts to avoid
traversing intermediate graph layers when they are unlikely to improve
the search. However, the remaining graph traversal still involves
irregular memory accesses to node vectors and adjacency lists. These
accesses can cause cache misses and memory stalls.

The central idea is to use the shortcut decision to control which data
is prefetched:

-   **If one or more intermediate layers will be skipped:** do not
    deliberately prefetch node data or adjacency lists needed
    exclusively by those layers.
-   **If a layer must be explored:** prefetch the data likely to be
    needed for that layer, such as the selected entry node and its
    adjacency list, and optionally a small number of likely neighbors.
-   **If the shortcut identifies a lower destination layer:** focus
    prefetching on the data required at that destination rather than
    data belonging only to bypassed layers.

The intended benefit is to reduce unnecessary prefetch traffic and cache
pollution while hiding some of the memory latency for data that the
search will actually use. This is a research hypothesis and must be
validated experimentally.

## 2. Motivation

Hierarchical graph indexes such as HNSW navigate a proximity graph
across multiple layers. Search performance can be limited by both
computation and memory access. SHG addresses unnecessary traversal and
expensive distance calculations through learned shortcuts and
hierarchical vector compression.

Prefetching is a possible complementary optimization. A conventional
prefetch strategy may fetch data before knowing whether the search will
visit the corresponding layer. If SHG later skips that layer, the
fetched data may be unused. Such requests can consume memory bandwidth
and cache capacity and may evict useful cache lines.

This proposal introduces **skip-aware prefetching**: use SHG's predicted
destination layer to decide which upcoming graph data should be fetched.

The core principle is:

> Do not prefetch data solely because it belongs to the next layer in
> the hierarchy. Prefetch data based on whether SHG expects to explore
> that layer.

## 3. Research Objective

The objective is to determine whether shortcut-aware, selective
prefetching improves SHG query latency without reducing search recall.

The study will investigate the following questions:

1.  Does skip-aware prefetching reduce cache misses and memory stalls
    compared with SHG without prefetching?
2.  Does it outperform unconditional prefetching by avoiding data
    associated only with skipped layers?
3.  Does cluster-aware selection of prefetch targets improve performance
    beyond prefetching only the destination entry node and its adjacency
    list?
4.  What are the effects of prefetch depth, cluster size, dataset size,
    embedding dimension, and query characteristics?
5.  Does the method preserve SHG's search behavior and recall?

## 4. Proposed Architecture

The proposed method has four main components.

### 4.1 Existing SHG Search

Run the original SHG search procedure. It uses the query, current entry
point, compressed representations, and learned shortcut mechanism to
determine which levels need to be traversed.

The proposed optimization should not replace the shortcut mechanism or
change the original neighbor-selection and stopping rules.

### 4.2 Skip-Aware Prefetch Controller

The controller consumes the shortcut decision, including the predicted
destination layer and the entry point that will be used to continue the
search.

It classifies data into two categories:

-   **Data associated only with skipped layers:** do not issue new
    prefetch requests for this data.
-   **Data required by a layer that will be explored:** consider
    prefetching it before demand access.

The controller must respect the actual control flow of the SHG
implementation. A layer's data may already be in cache, and a decision
to skip a layer does not remove data already fetched by demand accesses
or other mechanisms.

### 4.3 Selective Data Prefetching

When the destination layer is known, begin with a conservative strategy:

1.  Prefetch the destination entry node's vector data if it is not
    already likely to be available.
2.  Prefetch the entry node's adjacency-list cache lines when those
    addresses are available.
3.  Optionally prefetch a small number of likely neighbor records.
4.  Limit the number of outstanding or issued prefetches to avoid
    excessive bandwidth use.

Prefetching should occur early enough to overlap memory latency with
useful work. If the shortcut decision requires data that has not yet
been loaded, that decision cannot retroactively prevent those earlier
accesses. The optimization targets data needed after the decision.

### 4.4 Optional Cluster-Aware Target Selection

Clustering can improve the selection of likely future accesses. During
index construction, nodes at each relevant layer can be grouped using
graph connectivity, vector similarity, or a combination of both.

For each cluster, the index may store:

-   A node-to-cluster mapping.
-   A representative node, preferably an actual graph node such as a
    medoid when a real node is required.
-   Optional metadata about graph neighbors or frequently co-accessed
    nodes.

During search, if a layer will be explored, the current entry node
identifies a candidate cluster. The controller can prefetch the entry
node and a limited number of likely-to-be-accessed nodes from that
cluster.

A cluster centroid is a vector summary and is not necessarily a valid
graph node. It should not be used as a traversal entry point unless the
implementation explicitly defines how it maps to a real node. Clustering
is an optional extension; it should be evaluated separately from the
basic skip-aware prefetching mechanism.

## 5. Algorithm

Let the current layer be `x`, and let SHG predict a destination layer
`y`, where `y <= x`. Layers between `x` and `y` that the original
algorithm skips should not trigger prefetches for data used exclusively
by those layers.

### Case A: SHG skips one or more layers

Suppose the search is at layer 4 and SHG predicts that layers 3 and 2
can be skipped, continuing at layer 1.

The controller should:

1.  Accept SHG's destination layer and destination entry point.
2.  Avoid issuing prefetches for node data or adjacency lists needed
    exclusively by layers 3 and 2.
3.  Prefetch data for the actual destination entry point, where useful
    and where the address is known.
4.  Optionally prefetch the destination entry point's adjacency list and
    a small number of likely neighbor records.
5.  Continue the original SHG search from the destination layer.

### Case B: SHG does not skip the next layer

Suppose the search must explore layer 3.

The controller should:

1.  Determine the entry point that will be used at layer 3.
2.  Prefetch the entry node's vector data and relevant adjacency-list
    cache lines.
3.  Optionally prefetch a limited number of likely neighbor records.
4.  Continue normal graph traversal, allowing the search to consume
    prefetched data if the prediction was correct.

### Conceptual Pseudocode

``` text
function SHG_Search_With_Skip_Aware_Prefetch(query):
    current_layer = top_layer
    entry_node = initial_entry_node

    while current_layer >= base_layer:

        # Run the original SHG logic.
        # It determines the destination layer and entry point.
        decision = SHG_Shortcut_Decision(
            query, current_layer, entry_node
        )

        destination_layer = decision.destination_layer
        destination_entry = decision.entry_node

        if destination_layer < current_layer:
            # One or more intermediate layers are skipped.
            # Do not prefetch data needed exclusively by them.
            current_layer = destination_layer
            entry_node = destination_entry

            # Prepare only data likely to be used at the destination.
            prefetch_if_useful(entry_node.vector)
            prefetch_if_useful(entry_node.adjacency_list)

        else:
            # The layer must be explored.
            entry_node = destination_entry
            cluster = get_cluster(entry_node, current_layer)

            prefetch_if_useful(entry_node.vector)
            prefetch_if_useful(entry_node.adjacency_list)

            # Optional extension: prefetch a limited set of
            # likely neighbors based on graph/cluster metadata.
            prefetch_selected_neighbors(cluster, entry_node)

            # Preserve the original SHG layer-search behavior.
            entry_node = explore_layer_using_original_SHG(
                query, entry_node
            )

            if original_layer_search_finished():
                current_layer = current_layer - 1

    return original_SHG_result()
```

This is conceptual pseudocode, not a drop-in implementation. The paper's
exact shortcut and layer-search control flow must be followed. In
particular, a shortcut decision may be made after searching the current
layer, and a layer may visit multiple candidates. Prefetching must be
inserted at a point where the destination and useful target addresses
are available.

## 6. Implementation Considerations

### 6.1 Prefetch the right memory objects

A node's vector and adjacency list may be stored separately. Prefetching
one does not guarantee that the other is cached. Identify the actual
memory layout and determine which cache lines are needed by the search.

### 6.2 Do not prefetch an entire cluster by default

A cluster can contain many nodes, while a query may access only a few.
Begin with a small number of targets and tune the prefetch depth
experimentally.

### 6.3 Keep prefetching separate from search correctness

Prefetching should not change the candidate queue, distance
calculations, neighbor selection, shortcut decision, or stopping
conditions. It should only influence when data is requested from memory.
Under this design, recall should remain unchanged apart from any
unrelated nondeterminism or implementation effects.

### 6.4 Handle uncertainty and late decisions

The shortcut decision may not be available early enough to hide all
memory latency. Measure the time between issuing a prefetch and the
corresponding demand access. If the lead time is too short, the prefetch
may not improve latency.

### 6.5 Consider cache and bandwidth pressure

Too many prefetches can compete with demand loads and evict useful cache
lines. The controller should be evaluated with different prefetch depths
and should avoid duplicate or obviously unnecessary requests where
feasible.

## 7. Experimental Plan

Use the same datasets, query sets, recall targets, hardware, and search
parameters across the compared configurations.

### 7.1 Configurations

  -----------------------------------------------------------------------
  Configuration                       Purpose
  ----------------------------------- -----------------------------------
  HNSW                                Reference baseline

  Original SHG                        Measures the paper's existing
                                      optimization

  SHG + unconditional prefetching     Measures prefetching without skip
                                      awareness

  SHG + skip-aware prefetching        Tests the central proposal

  SHG + clustering + skip-aware       Tests whether clustering improves
  prefetching                         target selection
  -----------------------------------------------------------------------

The most important comparison for the central hypothesis is **SHG +
unconditional prefetching versus SHG + skip-aware prefetching**. This
isolates the value of conditioning prefetches on the shortcut decision.

### 7.2 Metrics

  -----------------------------------------------------------------------
  Metric                              Purpose
  ----------------------------------- -----------------------------------
  Query latency                       Measures end-to-end search
                                      performance

  L1, L2, and last-level cache misses Shows whether memory behavior
                                      changes

  CPU cycles per query                Helps assess the total cost

  Instructions per query              Detects controller and clustering
                                      overhead

  Memory bandwidth                    Detects excessive prefetch traffic

  Prefetch accuracy                   Fraction of prefetched data that is
                                      subsequently used, with a clearly
                                      defined measurement window

  Prefetch timeliness                 Whether data arrives before its
                                      demand access

  Recall@k                            Checks search quality

  Index memory overhead               Measures cluster metadata and other
                                      added storage
  -----------------------------------------------------------------------

Report results at comparable recall levels where relevant. Use repeated
runs and report variation, not just a single best measurement.

### 7.3 Ablation Studies

Evaluate the following variants separately:

1.  Prefetch only the destination entry node.
2.  Prefetch the destination entry node and its adjacency list.
3.  Add a small number of likely neighbors.
4.  Add vector-based clustering.
5.  Add graph-aware clustering.
6.  Compare different cluster sizes and prefetch depths.

This identifies which component provides the benefit and whether
additional metadata is worth its cost.

## 8. Expected Benefits and Risks

  -----------------------------------------------------------------------
  Aspect                  Expected effect         Risk or caveat
  ----------------------- ----------------------- -----------------------
  Unnecessary prefetch    May decrease for        Other mechanisms may
  traffic                 skipped layers          still fetch the data

  Cache pollution         May decrease            Useful prefetches can
                                                  still evict other data

  Memory stalls           May decrease for        Prefetches may be late
                          explored layers         or inaccurate

  Query latency           May improve             Controller overhead can
                                                  outweigh savings

  Recall                  Should remain unchanged Search code must not be
                          if traversal is         altered inadvertently
                          unchanged               

  Index overhead          Small if metadata is    Clustering adds storage
                          limited                 and construction work
  -----------------------------------------------------------------------

No performance improvement should be assumed in advance. The experiments
must establish whether the reduction in wasted prefetches is large
enough to outweigh the controller's overhead.

## 9. Proposed Research Contribution

The working title is:

**Skip-Aware Prefetching for Cache-Efficient Hierarchical Graph Search**

The main contribution is a prefetch controller that uses SHG's
layer-skipping decision to avoid deliberately fetching data needed
exclusively by bypassed layers and to prioritize data likely to be used
at layers that will be explored.

An optional second contribution is cluster-aware selection of prefetch
targets, using graph connectivity and/or vector similarity to identify
likely future accesses.

The central hypothesis is:

> By coupling prefetch decisions with SHG's predicted traversal path,
> the system can reduce unnecessary memory traffic and cache pollution
> while hiding part of the memory latency for data that the search will
> actually access.

This hypothesis should be tested against both original SHG and SHG with
unconditional prefetching, while measuring latency, cache behavior,
bandwidth, prefetch accuracy, and recall.

## 10. Recommended First Prototype

To keep the first implementation manageable and make the results
interpretable:

1.  Start from the original SHG implementation.
2.  Add prefetching only after the shortcut has identified the
    destination layer and entry point.
3.  Avoid prefetching data needed exclusively by skipped layers.
4.  Prefetch the destination entry node and its adjacency list when
    their addresses are available.
5.  Do not add clustering yet.
6.  Compare against original SHG and SHG with unconditional prefetching.
7.  If skip-aware prefetching helps, add graph-aware clustering as a
    separate experiment.

This staged approach tests the central idea before adding the complexity
of cluster construction and management.

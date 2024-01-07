def mse(pr_scores1, pr_scores2):
    """
    Calculate the Mean Squared Error (MSE) between two PageRank score dictionaries

    Arguments of the function:
    pr_scores1 (dict) -> A dictionary of PageRank scores
    pr_scores2 (dict) -> A dictionary of PageRank scores 

    Returns of the function:
    mse (float) -> The calculated MSE score
    """
    import numpy as np
    # Verify that both inputs are dictionaries
    if not isinstance(pr_scores1, dict) or not isinstance(pr_scores2, dict):
        # If one or both inputs has the wrong type rise a ValueError
        raise ValueError("Both inputs must be dictionaries.")
    # Initialize lists to store the scores from each dictionary
    scores1 = []
    scores2 = []
    # Iterate over the nodes in the first dictionary
    for node in pr_scores1:
        # Append the score from the first dictionary and the corresponding score from the second dictionary
        scores1.append(pr_scores1.get(node, 0))
        scores2.append(pr_scores2.get(node, 0))
    # Convert lists to numpy arrays for more efficient numerical operations
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)
    # Calculate the MSE score
    mse = np.mean((scores1 - scores2) ** 2)

    return mse

#-------------------------------------------------------------------------------------------------------------------------

def score_check(pr_scores1, pr_scores2, epsilon=0.0001):
    """
    Count the number of nodes where the difference in PageRank scores 
    between two dictionaries is less than the specified epsilon.

    Arguments of the function:
    pr_scores1 (dict) -> A dictionary of PageRank scores
    pr_scores2 (dict) -> A dictionary of PageRank scores 
    epsilon(int) -> The maximum difference that the two score can have in absolute value
                    by default is set to 0.0001

    Returns of the function:
    count (int) -> The percentage of nodes wich have a score differences less than epsilon
    """
    # Verify that both inputs are dictionaries
    if not isinstance(pr_scores1, dict) or not isinstance(pr_scores2, dict):
        # If one or both inputs has the wrong type rise a ValueError
        raise ValueError("Both inputs must be dictionaries.")
    # Initialize a counter
    count = 0  
    # Iterate over each key (node) in the first PageRank scores dictionary
    for key in pr_scores1:
        # Check if the node is present in the second dictionary
        if key in pr_scores2:
            # Compare the scores for the same node between dictionaries
            if abs(pr_scores1[key] - pr_scores2[key]) <= epsilon:
                # Increment the counter if the absolute value of the difference is less or equal than epsilon
                count += 1  
    return f'{(count/len(pr_scores1))*100} %'


#-------------------------------------------------------------------------------------------------------------------------

def extract_graph_data(graph, graph_name):
    '''
    Extract Graph Data

    Arguments of the function:
    graph (nx.Graph) -> A networkx graph 
    graph_name (str) -> A string representing the name of the graph

    Output:
    A dictionary containing all the requested data of the graph
    '''

    # Get the number of nodes in the graph
    num_nodes = len(graph.nodes())
    # Get the number of edges in the graph
    num_edges = len(graph.edges())
    # Calculate the density of the graph
    density = nx.density(graph)

    # Define an internal function to compute the degree distribution in bins
    def compute_degree_distribution(degrees):
        # Define the bins for degree distribution, with intervals of 25
        bins = range(0, max(degrees) + 25, 25)
        # Calculate the histogram of degrees
        hist, bin_edges = np.histogram(degrees, bins=bins)
        # Create a dictionary to map each degree range to its frequency
        distribution = {f"{int(bin_edges[i])}-{int(bin_edges[i+1])-1}": hist[i] for i in range(len(hist))}
        # Return the created sub-dictionary
        return distribution

    # Check if the graph is directed
    if graph.is_directed():
        # Collect the in-degrees for each node
        in_degrees = [d for n, d in graph.in_degree()]
        # Collect the out-degrees for each node
        out_degrees = [d for n, d in graph.out_degree()]
        # Combine in-degrees and out-degrees
        degrees = in_degrees + out_degrees
        # Compute the distribution of in-degrees
        in_degree_distribution = compute_degree_distribution(in_degrees)
        # Compute the distribution of out-degrees
        out_degree_distribution = compute_degree_distribution(out_degrees)
    else:
        # Collect the degrees for each node for an undirected graph
        degrees = [d for n, d in graph.degree()]
        # Compute the degree distribution
        degree_distribution = compute_degree_distribution(degrees)

    # Calculate the average degree
    avg_degree = np.mean(degrees)
    # Define the threshold for identifying hubs (95th percentile)
    hubs_threshold = np.percentile(degrees, 95)
    # Identify hubs as nodes with degrees above the threshold
    hubs = [n for n, d in graph.degree() if d > hubs_threshold]

    # Create a dictionary with the calculated data to return
    data = {
        "Graph Name": graph_name,
        "Number of Nodes": num_nodes,
        "Number of Edges": num_edges,
        "Graph Density": density,
        "Graph Type": "dense" if density > 0.5 else "sparse",
        "Average Degree": avg_degree,
        "Graph Hubs": hubs
    }

    # Add the degree distribution to the dictionary, based on whether the graph is directed or not
    if graph.is_directed():
        data["In-Degree Distribution"] = in_degree_distribution
        data["Out-Degree Distribution"] = out_degree_distribution
    else:
        data["Degree Distribution"] = degree_distribution

    # Return the dictionary with the graph data
    return data


#-------------------------------------------------------------------------------------------------------------------------



def node_contribution(graph, node, graph_name):
    '''
    Node Contribution Analysis

    Arguments:
    graph (nx.Graph) -> A networkx graph 
    node (int) -> The node for which we will make the analysis
    graph_name (str) -> The name of the graph.

    Output:
    A dictionary containing the centrality measures for 
    '''

    # Calculate the betweenness centrality 
    betweenness_centrality = nx.centrality.betweenness_centrality(graph)[node]
    # Calculate the PageRank value 
    pagerank = nx.pagerank(graph)[node]
    # Calculate the closeness centrality
    closeness_centrality = nx.centrality.closeness_centrality(graph, u=node)
    # Calculate the degree centrality 
    degree_centrality = nx.centrality.degree_centrality(graph)[node]

    # Return a dictionary with the centrality measures for the node
    return {
        "Node": node,
        "Graph": graph_name,
        "Betweenness Centrality": betweenness_centrality,
        "PageRank": pagerank,
        "Closeness Centrality": closeness_centrality,
        "Degree Centrality": degree_centrality,
    }


#-------------------------------------------------------------------------------------------------------------------------

def shortest_ordered_walk(graph, authors_a, a_1, a_n, N):
    '''
    This function finds the shortest ordered walk in a subgraph

    Argumnets of the function:
    graph (nx.Graph) -> A NetworkX graph
    authors_a (list of str) -> A sequence of authors that the path must traverse
    a_1 (str) -> Starting author
    a_n (str) -> Ending author
    N (int) -> The number of top papers to consider

    Return: 
    A dictionary with the shortest walk and papers crossed
    '''
    # Extract the top N papers in the graph based on degree centrality
    top_n_papers = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)[:N]

    # Create a subgraph that includes only these top N papers
    subgraph = graph.subgraph(top_n_papers).copy()

    # Check if both the start author and the end author are present in the subgraph
    if a_1 not in subgraph or a_n not in subgraph:
        return "One or more authors are not present in the graph."

    # Initialize a the list to store the sequence of authors in the shortest walk
    ordered_nodes = [a_1] + authors_a + [a_n]

    # Initialize a lists to store the shortest walk and the papers crossed
    shortest_walk = [] 
    papers_crossed = []

    def bfs_shortest_walk(graph, start, end):
        '''
        This function performs a breadth-first search to find the shortest path between two authors in the graph

        Argumnets of the function:
        graph (nx.DiGraph) -> A NetworkX graph
        start (str) -> Starting author
        end (str) -> Ending author
        
        Return: 
        A list containing the shortest path and the papers encountered
        '''
        # Initialize a queue for the BFS, starting with the start author
        queue = [(start, [start], [])]
        # Set to keep track of visited authors to avoid loops
        visited = set()

        # Loop to explore the graph using breadth-first search
        while queue:
            # Extract the current author, path so far, and papers encountered
            current, walk, papers = queue.pop(0)

            # If the current author is the end author, return the path and papers
            if current == end:
                return walk, papers

            # If the current author hasn't been visited, explore their connections
            if current not in visited:
                # Mark the current author as visited
                visited.add(current)
                # Iterate through each neighbor of the current author
                for neighbor in graph[current]:
                    # Extract paper information from the edge attributes
                    edge_attrs = graph[current][neighbor]
                    # Add the neighbor to the queue for further exploration
                    queue.append((neighbor, walk + [neighbor], papers + edge_attrs.get("titles", [])))

        # If no path is found return empty lists
        return [], []

    # Iterate through each pair of consecutive authors to find the shortest walk between them
    for i in range(len(ordered_nodes) - 1):
        # Find the shortest walk between the current pair of authors
        walk, papers = bfs_shortest_walk(subgraph, ordered_nodes[i], ordered_nodes[i + 1])

        # If no path exists between a pair, return a message
        if not walk:
            return "There is no such path."

        # Add the found path and papers to the respective lists
        shortest_walk.extend(walk[:-1])  # Exclude the last author as it will be included in the next pair
        papers_crossed.extend(papers)

    # Add the final author to complete the walk
    shortest_walk.append(a_n)

    # Return the shortest walk and the papers crossed in the walk
    return {"Shortest Walk": shortest_walk, "Papers Crossed": papers_crossed}


#-------------------------------------------------------------------------------------------------------------------------

def disconnecting_graphs(graph, authorA, authorB, N):
    '''
    Disconnecting Graphs

    Arguments of the function:
    graph (nx.Graph) -> A networkx graph object
    authorA, authorB (str) -> The nodes (authors) to disconnect
    N (int) -> Number of top nodes to consider based on degree centrality

    Output:
    Returns the initial subgraph, the components containing authorA and authorB after disconnection, and the number of edges removed
    '''

    # Select the top N authors based on degree centrality
    top_n_authors = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)[:N]
    # Create a subgraph including only the top N authors
    subgraph = graph.subgraph(top_n_authors).copy()
    # Store a copy of the initial subgraph for later comparison
    initial_subgraph = subgraph.copy()
    
    # Check if both authorA and authorB are present in the subgraph
    if authorA not in subgraph or authorB not in subgraph:
        print(f"One or both authors ({authorA}, {authorB}) not present in the graph.")
        return None, None, None, None

    def dfs(graph, visited, start):
        # Perform a depth-first search from a starting node
        stack = [start]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                # Add neighbors of the node to the stack, excluding already visited nodes
                stack.extend(set(graph.neighbors(node)) - visited)
        return visited

    def min_edge_cut_between_subgraphs(graph, nodes_A, nodes_B):
        # Calculate the minimum edge cut required to separate two sets of nodes
        min_edge_cut = []
        # Perform DFS to find all nodes connected to nodes_A
        visited_A = dfs(graph, set(), next(iter(nodes_A)))
        for node_A in visited_A:
            for neighbor_B in nodes_B:
                if graph.has_edge(node_A, neighbor_B):
                    # Add the edge to the min-edge cut if it connects nodes_A to nodes_B
                    min_edge_cut.append((node_A, neighbor_B))
        return min_edge_cut

    # Calculate the min-edge cut needed to disconnect authorA from authorB
    min_edge_cut = min_edge_cut_between_subgraphs(subgraph, [authorA], [authorB])
    # Remove the edges from the subgraph to disconnect the authors
    subgraph.remove_edges_from(min_edge_cut)
    # Find connected components in the modified subgraph
    components = list(nx.connected_components(subgraph))
    # Identify the components containing authorA and authorB
    G_a = next((comp for comp in components if authorA in comp), None)
    G_b = next((comp for comp in components if authorB in comp), None)

    # Return the initial subgraph, components containing authorA and authorB, and the number of edges removed
    return initial_subgraph, G_a, G_b, len(min_edge_cut)


#-------------------------------------------------------------------------------------------------------------------------

def extract_communities(graph, N, paper_1, paper_2):
    """
    Extracts communities from a graph using Girvan-Newman algorithm
    
    Arguments of the function:
    graph (nx.Graph) -> The graph from which to extract communities
    N (int) -> Number of top nodes to consider based on degree centrality
    paper_1 (str) -> The first paper
    paper_2 (str) -> The second paper

    Returns:
    A tuple containing the subgraph with top N nodes, number of edges removed, 
    list of communities, and a boolean indicating if paper_1 and paper_2 are in the same community
    """

    def edge_to_remove_directed(graph):
        # Initialize the minimum weight to infinity and minimum edge to None
        min_weight = float('inf')
        min_edge = None
        # Iterate through all edges in the graph
        for edge in graph.edges(data=True):
            # Check if the weight of the current edge is less than the minimum weight found so far
            if edge[2]['weight'] < min_weight:
                # Update minimum weight and the edge associated with it
                min_weight = edge[2]['weight']
                min_edge = edge[:2]
        # Return the edge with the minimum weight
        return tuple(min_edge)

    def edge_to_remove_undirected(graph):
        # Find the edge with the highest betweenness centrality in the undirected graph
        edge_betweenness = calculate_edge_betweenness(graph)
        return max(edge_betweenness, key=edge_betweenness.get)

    def calculate_edge_betweenness(graph):
        # Initialize an empty dictionary for edge betweenness centrality
        edge_betweenness = {}
        # Iterate through all edges in the graph
        for edge in graph.edges():
            # Calculate betweenness centrality for each edge
            edge_betweenness[edge] = calculate_edge_betweenness_centrality(graph, edge)
        # Return the dictionary containing edge betweenness centrality for all edges
        return edge_betweenness

    def calculate_edge_betweenness_centrality(graph, edge):
        # Create a mapping from each node to its neighbors excluding the other node in the edge
        node_to_neighbors = {v: set(graph.neighbors(v)) - {u} for u, v in graph.edges()}
        # Initialize a dictionary to track the shortest paths starting from each node
        node_to_shortest_paths = {node: {node} for node in graph.nodes()}

        # Perform Breadth-First Search to find shortest paths
        # Initialize a queue with the target node of the edge and a visited set
        queue = [edge[1]]
        visited = set()
        visited.add(edge[1])
        # Process nodes in the queue
        while queue:
            current_node = queue.pop(0)
            # Iterate through the neighbors of the current node
            for neighbor in node_to_neighbors.get(current_node, []):
                if neighbor not in visited:
                    # Mark the neighbor as visited and add it to the queue
                    visited.add(neighbor)
                    queue.append(neighbor)
                    # Update the shortest paths to include paths through the current node
                    node_to_shortest_paths[neighbor] = set.union(node_to_shortest_paths[neighbor], node_to_shortest_paths[current_node], {neighbor})

        # Calculate betweenness centrality for the given edge
        betweenness_centrality = 0
        # Iterate through all nodes in the graph
        for node in graph.nodes():
            # Exclude the nodes that are part of the edge
            if node != edge[0] and node != edge[1]:
                # Check if the edge is part of the shortest paths of the node
                for path in node_to_shortest_paths.get(node, []):
                    if edge[0] in path and edge[1] in path:
                        # Increment the betweenness centrality for each shortest path the edge is part of
                        betweenness_centrality += 1 / len(node_to_shortest_paths[node])

        # Return the calculated betweenness centrality for the edge
        return betweenness_centrality

    def girvan_newman_directed(graph):
        # Implement Girvan-Newman algorithm for community detection in a directed graph
        while nx.number_weakly_connected_components(graph) == 1:
            edge_to_remove_value = edge_to_remove_directed(graph)
            graph.remove_edge(*edge_to_remove_value)
        return list(nx.weakly_connected_components(graph))

    def girvan_newman_undirected(graph):
        # Implement Girvan-Newman algorithm for community detection in an undirected graph
        while nx.number_connected_components(graph) == 1:
            edge_to_remove_value = edge_to_remove_undirected(graph)
            graph.remove_edge(*edge_to_remove_value)
        return list(nx.connected_components(graph))

    # Select the top N papers based on degree centrality
    top_n_papers = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)[:N]
    
    # Create a subgraph with only the top N papers
    subgraph = graph.subgraph(top_n_papers).copy()
    initial_subgraph = subgraph.copy()

    # Check if both papers are present in the subgraph
    if paper_1 not in subgraph or paper_2 not in subgraph:
        print(f"One or both papers ({paper_1}, {paper_2}) not present in the graph.")

    # Perform Girvan-Newman community detection
    if graph.is_directed():
        communities = girvan_newman_directed(subgraph)
    else:
        communities = girvan_newman_undirected(subgraph)
        
    # Check if Paper_1 and Paper_2 belong to the same community
    same_community = any([paper_1 in community and paper_2 in community for community in communities])

    # Calculate the minimum number of edges to be removed
    num_edges_to_remove = len(initial_subgraph.edges()) - len(subgraph.to_undirected().edges())
    
    return subgraph, num_edges_to_remove, communities, same_community


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
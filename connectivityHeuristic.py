# Algorithm 6.1 in the TSP book; parametric family
import numpy as np

def paramConnHeuristic(x):
    # Note that x is np.ndarray(dtype=np.double)
    # Each element of x denotes... (0, 1), (0, 2), ..., (0, n-1), ..., (n-2, n-1).
    n = int(np.sqrt(2*len(x))) + 1
    print(n)
    assert(n * (n-1) == 2 * len(x))

    parent = list(range(n))
    print(parent)

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j))

    priority_indices = np.argsort(-x)  # 큰 값부터 인덱스 정렬
    for idx in priority_indices:
        print(idx)  # 값이 아니라 인덱스 출력

    
    # 1. sort x
    return []
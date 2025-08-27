# Algorithm 6.3 (Actually, Algorithm 6.1 and 6.2 are the same too) 
# in the TSP book; parametric family
import numpy as np
from utils import numVertex, delta, corrIndex, subtourCut

# find cut : cut함수는 1차원 fractional solution np.array(dtype=np.double)을 받아서, 
# 무엇을 반환하냐면 : List of (indices, vals, rhs). 부등호는 무조건 greater than.
# indices - 얘는 np.ndarray(dtype=np.int32) (1차원. coefficient 명시할 indices)
# vals - 얘는 np.ndarray(dtype=np.double) (1차원. size는 indices와 같음. coefficient 명시)
# rhs - 얘는 np.double : right hand side. 

subtour_all = []

def paramConnHeuristic(x):
    # Note that x is np.ndarray(dtype=np.double)
    # Each element of x denotes... (0, 1), (0, 2), ..., (0, n-1), ..., (n-2, n-1).
    n = numVertex(x)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j))

    sets = [] 
    for i in range(n): sets.append({i})

    def find(set, i):
        for l in range(len(sets)):
            if (i in sets[l]): return l
        assert False
        return -1

    subtours = []

    priority_indices = np.argsort(-x)  # 큰 값부터 인덱스 정렬
    for idx in priority_indices:
        i, j = edges[idx]
        # i를 포함하는 set과 j를 포함하는 set의 index를 출력
        i1 = find(sets, i)
        if (j in sets[i1]): continue # i와 j가 같은 component라면? nothing to do.
        j1 = find(sets, j)

        if (i1 > j1): i1, j1 = j1, i1 # 원활한 진행을 위해 순서 바꾸기
        sets[i1].update(sets.pop(j1))
        if (sets[i1] == set(range(n))): break # terminal condition

        if (delta(x, sets[i1]) < 2) and (sets[i1] not in subtours) and ((set(range(n)) - sets[i1]) not in subtours):
            if len(sets[i1]) > len(set(range(n)) - sets[i1]):
                subtours.append(set(range(n)) - sets[i1])    
            else:
                subtours.append(sets[i1].copy())    

    cuts = []
    for subtour in subtours:
        if subtour in subtour_all: continue
        cuts.append(subtourCut(n, subtour))
        subtour_all.append(subtour)

    print(str(len(cuts)) + " subtour inequalities added.")
    return cuts
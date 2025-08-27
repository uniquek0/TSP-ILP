import numpy as np
import itertools

def numVertex(x):
    return int(np.sqrt(2*len(x))) + 1

def corrIndex(n, edge):
    sorted_list = sorted(edge)
    i = sorted_list[0]
    j = sorted_list[1]
    m = i*(2*n-i-1)//2 + (j-i-1)
    #assert (corrTuple(n, m) == set(edge))
    #print("TRUE!")
    return m
    '''
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append({i, j})
    return edges.index(set(edge))
    '''

def corrTuple(n, m):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append({i, j})
    return edges[m]

def sumInsideEdges(x, n, ind):
    assert (numVertex(x) == n)
    sum = 0
    for subset in itertools.combinations(ind, 2):
        sum += x[corrIndex(n, subset)]
    return sum 

def between(x, ind1, ind2):
    n = numVertex(x)
    sum = 0
    for i in ind1:
        for j in ind2:
            sum += x[corrIndex(n, {i, j})]
    return sum 

def delta(x, ind): # ind : set
    n = numVertex(x)
    m = len(ind)
    assert (m > 0) and (m < n)
    if (2 * m < n): return 2 * m - 2 * sumInsideEdges(x, n, ind)
    return 2 * (n-m) - 2 * sumInsideEdges(x, n, set(range(n)) - ind)


def combCut(n, comb):
    # comb : (handle, list of teeth). 각각의 tooth는 set.
    handle, teeth = comb
    cutdic = dict()
    cutrhs = ((3 * len(teeth)+2)//2) - len(handle) 
    for subset in itertools.combinations(handle, 2):
        cutdic[corrIndex(n, subset)] = -1
    for tooth in teeth:
        cutrhs -= len(tooth)
        for subset in itertools.combinations(tooth, 2):
            cutdic[corrIndex(n, subset)] = cutdic.get(corrIndex(n, subset), 0) - 1
    cutinds = np.array(list(cutdic.keys()), dtype=np.int32)
    cutcoeffs = np.array(list(cutdic.values()), dtype=np.double)
    checkCut(cutinds, cutcoeffs, np.double(cutrhs))
    return (cutinds, cutcoeffs, np.double(cutrhs))


def pathCut(n, star):
    # star : (list of handles, list of teeth). 각각의 handle이나 tooth는 set.
    handles, teeth = star
    cutdic = dict()
    cutrhs = ((len(handles) + 2) * len(teeth) + len(handles))//2
    print(len(handles), len(teeth), cutrhs)
    for handle in handles:
        cutrhs -= len(handle)
        for subset in itertools.combinations(handle, 2):
            cutdic[corrIndex(n, subset)] = cutdic.get(corrIndex(n, subset), 0) - 1
    for tooth in teeth:
        cutrhs -= len(tooth)
        for subset in itertools.combinations(tooth, 2):
            cutdic[corrIndex(n, subset)] = cutdic.get(corrIndex(n, subset), 0) - 1 
    cutinds = np.array(list(cutdic.keys()), dtype=np.int32)
    cutcoeffs = np.array(list(cutdic.values()), dtype=np.double)
    checkCut(cutinds, cutcoeffs, np.double(cutrhs))
    return (cutinds, cutcoeffs, np.double(cutrhs))

def subtourCut(n, subtour):
    # subtour : set.
    cutinds = []
    cutrhs = 1 - len(subtour)
    for subset in itertools.combinations(subtour, 2):
        cutinds.append(corrIndex(n, subset))
    cutcoeffs = np.full(len(cutinds), -1)
    cutinds = np.array(cutinds, dtype=np.int32)
    checkCut(cutinds, cutcoeffs, np.double(cutrhs))
    return (cutinds, cutcoeffs, np.double(cutrhs))


def checkCut(cutinds, cutcoeffs, cutrhs):
    pass
    tour = [0, 518, 38319, 38357, 38282, 38240, 38199, 38157, 38114, 38070, 38025, 37979, 37932, 37884, 37835, 37849, 38430, 38431, 38501, 38595, 37805, 37734, 37682, 37629, 37575, 37520, 37464, 37407, 37349, 37290, 37230, 37169, 37107, 37044, 36980, 36915, 36849, 36782, 36714, 36645, 36575, 36361, 36285, 36210, 36134, 36057, 35979, 35900, 35659, 35574, 35490, 35405, 35319, 35232, 35144, 35055, 34965, 34874, 34782, 34689, 34595, 34500, 34404, 34307, 34209, 34110, 33604, 33603, 33909, 30573, 30572, 33704, 30700, 30675, 30807, 30934, 30933, 31059, 27608, 27584, 27735, 5240, 5130, 5496, 27279, 27125, 26970, 26814, 26657, 26499, 26340, 26180, 26019, 26056, 31434, 31557, 31679, 31814, 31933, 31920, 32039, 32157, 32274, 32390, 32505, 32619, 32732, 32844, 32956, 33065, 33066, 33282, 24075, 23835, 23660, 23484, 23307, 23129, 22950, 22770, 22589, 22407, 22224, 22040, 21855, 21669, 21482, 21294, 21105, 20915, 20724, 20743, 24182, 24183, 24525, 24695, 20555, 20339, 20364, 25032, 25199, 25366, 25530, 20174, 19950, 19754, 19557, 19359, 19160, 18960, 18759, 18557, 18354, 18150, 17945, 17739, 17532, 17324, 17115, 16905, 16694, 16482, 16269, 16055, 15840, 15624, 14312, 14084, 13860, 13635, 13409, 13182, 12954, 12725, 12495, 12264, 12032, 11799, 11565, 11330, 11094, 11108, 14532, 15189, 15244, 15026, 14750, 10873, 10619, 10380, 10140, 9899, 9657, 9414, 9170, 8925, 8679, 8432, 8184, 7935, 7685, 7434, 7182, 6929, 6675, 5652, 5651, 5908, 5907, 3558, 3549, 3283, 3014, 2745, 2475, 2204, 1932, 1659, 1385, 1110, 834, 1106, 39050, 39045, 39039, 39032, 39024, 39015, 4334, 4080, 4344, 4607, 4981, 28034, 28182, 28464, 38994, 28608, 28475, 28751, 38969, 28893, 28764, 28907, 29058, 30282, 30149, 30015, 29880, 29933, 35739, 29799, 29607, 29469, 29330, 29190, 29315, 38940, 38924, 38907, 38889, 38870, 38850, 38829, 38807, 38784, 38711, 38682, 36476, 36432, 36546, 38656, 38735, 38570, 38532, 38561, 39057, 832, 833, 278]
    sum = 0
    for v in tour:
        indices = np.where(cutinds == v)[0]
        if indices.size > 0:
            assert(indices.size == 1)
            s = indices[0]
            sum += cutcoeffs[s]
    
    print(sum, cutrhs)
    assert(sum >= cutrhs)


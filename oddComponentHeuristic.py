from utils import numVertex, delta, corrIndex, between, combCut, pathCut, subtourCut

# odd-component Heuristic

# find cut : cut함수는 1차원 fractional solution np.array(dtype=np.double)을 받아서, 
# 무엇을 반환하냐면 : List of (indices, vals, rhs). 부등호는 무조건 greater than.
# indices - 얘는 np.ndarray(dtype=np.int32) (1차원. coefficient 명시할 indices)
# vals - 얘는 np.ndarray(dtype=np.double) (1차원. size는 indices와 같음. coefficient 명시)
# rhs - 얘는 np.double : right hand side. 

def remove_intersecting_pairs_and_get_common_elems(sets_list):
    from collections import defaultdict

    elem_to_indices = defaultdict(list)
    for i, s in enumerate(sets_list):
        for e in s:
            elem_to_indices[e].append(i)

    to_remove = set()
    common_elems = set()

    # 2개 집합에 걸치는 원소가 있으면,
    # 그 집합 인덱스와 그 원소를 저장
    for e, indices in elem_to_indices.items():
        if len(indices) == 2:
            to_remove.update(indices)
            common_elems.add(e)

    remaining_sets = [s for i, s in enumerate(sets_list) if i not in to_remove]

    return remaining_sets, common_elems

def fractionalComponents(x):

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

    for idx in range(len(x)):
        if ((x[idx] == 0.0) or (x[idx] == 1.0)): continue
        i, j = edges[idx]
        # i를 포함하는 set과 j를 포함하는 set의 index를 출력
        i1 = find(sets, i)
        if (j in sets[i1]): continue # i와 j가 같은 component라면? nothing to do.
        j1 = find(sets, j)

        if (i1 > j1): i1, j1 = j1, i1 # 원활한 진행을 위해 순서 바꾸기
        sets[i1].update(sets.pop(j1))
    
    for i in range(len(sets)-1, -1, -1):
        if (len(sets[i]) == 1): sets.pop(i)
    #print(sets)

    return sets


def oddCompHeuristic(x):
    # Note that x is np.ndarray(dtype=np.double)
    # Each element of x denotes... (0, 1), (0, 2), ..., (0, n-1), ..., (n-2, n-1).
    n = numVertex(x)

    sets = fractionalComponents(x)
    combs = []
    for set in sets:
        if (len(set) == 1): continue
        # set 바깥에서 여기로 들어오는 친구가 있는지 확인
        teeths = []
        for i in range(n):
            if i in set: continue
            for j in set:
                if (x[corrIndex(n, {i, j})] == 1.0):
                    teeths.append([j, i])
        if (len(teeths) % 2 == 0): continue

        # 만약 teeth 두개가 같은 것을 포함하고 있다면 set으로 옮긴다.
        r, c = remove_intersecting_pairs_and_get_common_elems(teeths)
        if (len(r) < 3): continue
        combs.append((set | c, r))

    cuts = [combCut(n, comb) for comb in combs]

    print(str(len(cuts)) + " comb inequalities added.")

    # Path Inequality를 추가하고 싶다...
    numPaths = 0
    for comb in combs:
        handle, teeth = comb
        # tooth는 길이 2짜리인 list. 
        starTeeth = []
        maxList = []
        val = 0
        newhandle = handle.copy()
        valid = True
        for tooth in teeth:
            j, i = tooth[0], tooth[1]
            newhandle.add(i) # newhandle이 더 큰 handle

            # 바깥쪽에서 tooth마다 max 하나씩 잡아온다.
            maxval = 0
            maxind = -1
            for k in range(n): 
                if k in handle: continue
                inTooth = False
                for tooth2 in teeth:
                    if k in tooth2: inTooth = True
                if (inTooth) : continue
                if k in maxList: continue
                if (x[corrIndex(n, {i, k})] > maxval):
                    maxval = x[corrIndex(n, {i, k})]
                    maxind = k
            starTeeth.append({j, i, maxind})
            if (maxind == -1) : valid = False
            maxList.append(maxind)
            val += 2 - 2 * maxval
        if (not valid) : continue
        if (val > 1.99): continue
        print(comb)
        print(val, ([handle, newhandle], starTeeth))

        lhs = delta(x, handle) + delta(x, newhandle)
        for starTooth in starTeeth:
            lhs += delta(x, starTooth)
        print(lhs, len(starTeeth))
        assert(lhs < 4 * len(starTeeth) + 2.0)
        #cuts.append(pathCut(n, ([handle, newhandle], starTeeth)))
        numPaths += 1

        '''
        for i in range(n):
            if i in newhandle: continue
            if i in maxList: continue
            newval = val + 2 - 2 * between(x, newhandle, {i})
            if (newval < 1.99): 
                #print(([handle, newhandle | {i}], starTeeth))   
                cuts.append(pathCut(n, ([handle, newhandle | {i}], starTeeth)))
                numPaths += 1       
        '''
    print(str(numPaths) + " path inequalities added.")   
    return cuts
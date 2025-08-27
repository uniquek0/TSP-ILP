from utils import numVertex, delta, corrIndex, between, combCut, pathCut, subtourCut

def combExtendHeuristic(x, comb, d):
    # d value가 LHS. (x(delta)들의 합)
    n = numVertex(x)
    handle, teeth = comb
    newhandle = handle.copy()
    
    valid = True
    stars = []
    val = d
    while valid:
        maxList = []
        starTeeth = []
        for tooth in teeth:
            maxval = 0
            maxind = -1
            newhandle = handle | tooth
            for k in range(n): 
                if k in handle: continue
                if k in tooth: continue 
                if k in maxList: continue
                b = between(x, {k}, tooth)
                if (b > maxval):
                    maxval = b
                    maxind = k
            if maxind == -1:
                valid = False
            starTeeth.append(tooth | {maxind})
            maxList.append(maxind)
            val += 2 - 2 * maxval
        
        handle = newhandle
        newhandle = handle.copy()
    return stars
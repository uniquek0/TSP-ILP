'''
import math

from numpy import array
from lpsolvers import solve_lp
print("1")
'''
#from lk_heuristic.utils.solver_funcs import solve

import sys
import tsplib95
import numpy as np
from highspy import Highs, HighsLp, HighsSparseMatrix, MatrixFormat
from scipy.sparse import csr_matrix

import cplex
import connectivityHeuristic

SOLVER = "CPLEX"
CUTFUNCTIONS = [connectivityHeuristic.paramConnHeuristic]

def create_tsp_LP_relaxation(problem):
    nodes = list(problem.get_nodes())
    n = len(nodes)
    edges = []
    edge_idx = {}
    c = []

    # 1. Create variable for each undirected edge
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            dist = problem.get_weight(nodes[i], nodes[j])
            edges.append((i, j))
            edge_idx[(i, j)] = idx
            c.append(dist)
            idx += 1
    num_vars = len(c)

    # 2. Degree constraints: each node must have degree 2
    A = []
    b = []
    for i in range(n):
        row = [0] * num_vars
        for j in range(n):
            if i < j and (i, j) in edge_idx:
                row[edge_idx[(i, j)]] = 1
            elif j < i and (j, i) in edge_idx:
                row[edge_idx[(j, i)]] = 1
        A.append(row)
        b.append(2)  # each node must have degree 2

    A = np.array(A, dtype=np.double)
    b = np.array(b, dtype=np.double)
    c = np.array(c, dtype=np.double)

    lower_bounds = np.array(np.zeros(num_vars), dtype=np.double)
    upper_bounds = np.array(np.ones(num_vars), dtype=np.double)  # LP relaxation → x ∈ [0,1]

    return c, A, b, lower_bounds, upper_bounds, edges

def make_cplex_model(c, A, b, lower_bounds, upper_bounds):
    model = cplex.Cplex()
    model.set_log_stream(None)  # 로그 출력 끄기
    model.set_results_stream(None)
    model.set_warning_stream(None)

    # 목적함수 방향 (최소화)
    model.objective.set_sense(model.objective.sense.minimize)

    num_vars = len(c)
    num_rows = len(b)

    # 변수 추가: 계수, lower_bounds, upper_bounds
    model.variables.add(obj=c.tolist(),
                        lb=lower_bounds.tolist(),
                        ub=upper_bounds.tolist())

    # 제약조건 추가 (Ax = b)
    for i in range(num_rows):
        # A[i,:]에서 0 아닌 항목만 SparsePair로 넘김
        indices = np.nonzero(A[i])[0].tolist()
        values = A[i, indices].tolist()
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=indices, val=values)],
            senses=["E"],  # E = equality (=)
            rhs=[b[i]]
        )
    
    return model

def solve_cplex_model(model):
    # 최적화 실행
    model.solve()

    status = model.solution.get_status()
    status_str = model.solution.get_status_string(status)
    obj_val = model.solution.get_objective_value()
    sol = model.solution.get_values()
    iteration_count = model.solution.progress.get_num_iterations()

    print("model status:", status_str)

    return obj_val, np.array(sol, dtype=np.double), iteration_count

def solve_with_cplex(c, A, b, lower_bounds, upper_bounds):
    return solve_cplex_model(make_cplex_model(c, A, b, lower_bounds, upper_bounds))

def solve_with_highs(c, A, b, lower_bounds, upper_bounds):
    # Create HiGHS solver instance
    highs = Highs()
    highs.setOptionValue('output_flag',False)

    # Convert dense A to CSR
    A_sparse = csr_matrix(A).tocsc()
    
    # Manually extract CSR components for HighsSparseMatrix
    a_matrix = HighsSparseMatrix()
    a_matrix.num_col_ = A_sparse.shape[1]
    a_matrix.num_row_ = A_sparse.shape[0]
    a_matrix.start_ = A_sparse.indptr.tolist()
    a_matrix.index_ = A_sparse.indices.tolist()
    a_matrix.value_ = A_sparse.data.tolist()

    # Set up the LP using HighsLp
    lp = HighsLp()
    lp.num_col_ = len(c)
    lp.num_row_ = len(b)
    lp.col_cost_ = c
    lp.col_lower_ = lower_bounds
    lp.col_upper_ = upper_bounds
    lp.row_lower_ = b
    lp.row_upper_ = b
    lp.a_matrix_ = a_matrix

    # Pass the LP to the solver
    highs.passModel(lp)

    # Run the solver
    highs.run()

    # Extract results
    model_status = highs.getModelStatus()
    primal_solution = highs.getSolution()
    objective_value = highs.getInfo().objective_function_value
    iteration_count = highs.getInfo().simplex_iteration_count

    print("model status : ", highs.modelStatusToString(model_status))

    return objective_value, np.array(list(primal_solution.col_value), dtype=np.double), iteration_count

def main(args):
    if len(args) == 0: print("Please provide a TSPLIB file."); return
    for arg in args: 
        print("Processing " + arg); 
        problem = tsplib95.load(arg)
        print(f"Loaded problem with {len(list(problem.get_nodes()))} nodes.")
        main_loop_cplex(problem) if SOLVER == "CPLEX" else main_loop_highs(problem)

def main_loop_cplex(problem):
    c, A, b, lb, ub, edges = create_tsp_LP_relaxation(problem)
    model = make_cplex_model(c, A, b, lb, ub)
    while True :
        obj, sol, itr = solve_cplex_model(model)
        print(f"LP relaxation lower bound: {obj:.2f}")
        print(f"Iteration count : {itr}")

        # find cut : cut함수는 1차원 fractional solution np.array(dtype=np.double)을 받아서, 
        # 무엇을 반환하냐면 : List of (indices, vals, rhs). 부등호는 무조건 greater than.
        # indices - 얘는 np.ndarray(dtype=np.int32) (1차원. coefficient 명시할 indices)
        # vals - 얘는 np.ndarray(dtype=np.double) (1차원. size는 indices와 같음. coefficient 명시)
        # rhs - 얘는 np.double : right hand side. 
        cutList = []
        for f in CUTFUNCTIONS: cutList += f(sol)
        if len(cutList) == 0: break
        for cut in cutList:
            cutinds, cutcoeffs, cutrhs = cut
            model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=cutinds, val=cutcoeffs)],
                senses=["G"],
                rhs=[cutrhs]
            )

def main_loop_highs(problem):
    c, A, b, lb, ub, edges = create_tsp_LP_relaxation(problem)
    while True :
        obj, sol, itr = solve_with_highs(c, A, b, lb, ub)
        print(f"LP relaxation lower bound: {obj:.2f}")
        print(f"Iteration count : {itr}")
        break

if __name__ == "__main__":
    main(sys.argv[1:])
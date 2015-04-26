# simplex.jl
The function simplex() performs the simplex method (http://en.wikipedia.org/wiki/Simplex_algorithm) on a given linear programming problem in canonical form (or canonical problem). For now, the linear programming problem is initialized in the first few lines of the body of simplex(). 

A canonical problem consists of an m by n matrix A, an m-dimensional vector b, and a linear objective function (or cost function) c: R^{n} -> R, which is often represented as an n-dimensional vector. An n-dimensional vector u that satisfies the matrix equation Ax = b is called a *feasible solution*, and a feasible solution that also minimizes the objective function c(x) over all feasible solutions is called an *optimal solution*. A feasible (optimal) solution u is said to be a *basic feasible (optimal) solution* if the column vectors of A corresponding to the non-zero entries of u are linearly independent. 

The simplex method has two phases: in phase I, we find a basic feasible solution to the given canonical problem (or we show that no such solution exists). In phase II, we take a basic feasible solution to the given canonical problem and either (i) show the input solution is optimal, (ii) show there exists no optimal solution to the given canonical problem, or (iii) produce a new basic feasible solution with cost strictly less than the input feasible solution. If the third case, we take the output basic feasible solution and run it through phase II again. We iterate until we arrive at an optimal solution or show that no optimal solution exists.

The overall structure of the algorithm is interesting. Phase I actually calls Phase II on a modified version of the input linear programming problem, and the first iteration of Phase II takes as input the basic feasible solution found in Phase I. 

Note 1: It can be proved that there are only a finite number of basic feasible solutions, so the algorithm is guaranteed to stop... eventually.

Note 2: Technically we must impose a few additional constraints on the input canonical problem in order that the simplex method work, but they are generally satisfied in practice. In particular, we require: (i) that m > n; (ii) that A has rank m, which is to say that all of A's row vectors are linearly independet; and (iii) that the vector b cannot be written as a linear combination of fewer than m columns of A.

-----

%%%  Project ideas for Recurse Center pairing interview  %%% 

1.  One can show that any basic feasible solution to a canonical problem that satisfies the three conditions imposed in Note 2 will necessarily have m nonzero entries. Thus, Phase II should both take as input and return as output a vector with precisely m nonzero entries. However, phaseII() will occasionally return a vector that has fewer than m nonzero entries due to floating point approximations. This becomes problematic when iterating phaseII() on its own output vector since the body of phaseII() involves matrix computations that throw an error if the input vector has fewer than m nonzero entries. (In particular, on line 89 taking the inverse of AB will fail due to the specification of AB when the input vector x has fewer than m nonzero entries.) I'd like to implement a robust method for identifying and mitigating such floating point errors so that phaseII() can be guaranteed to return a vector with precisely m nonzero entries.

2.  Create an 'LPP' ('Linear Programming Problem') type and refactor simplex(), phaseI() and phaseII() for clarity.

3.  Optimize code for performance/make more idiomatic. What I have in mind right now is mostly figuring out which of my variables I can declare as constants...

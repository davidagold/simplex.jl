function simplex() 		# (A::Array{Float64, 2}, b::Array{Float64, 1}, c::Array{Float64, 1})
	# Initialize the constraint matrix A, constraint vector b, objective function c
	# In the future these will be passed as arguments to simplex()
	A = Float64[4 2 5 15 4; 1 4 5 1 2; 1 0 3 1 11]
	b = Float64[17, 9, 16]
	c = Float64[15, 4, 10, 11, 27]

	println("We first find a basic feasible solution to the given problem (or show that none exists).")

	# Perform Phase I on given problem
	x, possible = phaseI(A, b)
	possible == false && return "There are no basic feasible solutions."

	println("Ax = ", A*x)
	# println("Ax - b = ", *(A, x) - b)

	println()
	println("We now perform Phase II on the basic feasible solution x.")

	println("Step0 x = ", x, " with c(x) = ", (c'x)[1])

	# Perform Phase II on x
	for i in 1:20
		x, optimal, cost = phaseII(x, A, b, c)
		if optimal == true
			println("An optimal solution is ", x, " with c(x) = ", cost)
			println("Ax = ", A*x)
			break
		elseif cost == -Inf
			println("There is no optimal solution.")
		else
			println("Step", i, " x = ", x, " with c(x) = ", cost)
		end
	end
end

#Simplex method Phase I
function phaseI(A::Array{Float64, 2}, b::Array{Float64, 1})
	n = length(A[1,:])
	m = length(b)
	Aprime = hcat(A, eye(m))
	c = vcat(zeros(n), ones(m))
	x = vcat(zeros(n), b)

	println("Step0 x = ", x)

	for i in 1:50
		x, optimal, cost = phaseII(x, Aprime, b, c)
		println("cost = ", cost)
		if optimal == true
			if cost == 0
				println("A basic feasible solution is x = ", x)
				return x[1:n, 1], true
			else
				println("There are no basic feasible solutions.")
				return x[1:n, 1], false
			end
		end
		println("Step", i, " x = ", x)
 	end
end


#Simplex method Phase II
function phaseII(x::Array{Float64,1}, A::Array{Float64, 2}, b::Array{Float64, 1}, c::Array{Float64, 1})
	case1 = true
	subcase1 = true
	n = length(x)
	m = length(b)
	s = Int64
	t = Array(Float64, 1)
	lambdaset = Array(Float64, 1)
	lambda = Float64
	xnew = zeros(x)

	# Set the basis set of indices
	basis::Array{Int, 1} = find(x)

	# Set the m x m basis matrix AB
	AB::Array{Float64, 2} = [ A[i, j] for i in 1:m, j in basis ]

	# Set the M-dimensional basis vector xB
	xB::Array{Float64, 1} = [ x[j] for j in basis ]

	# Set the M-dimensional cost vector cB
	cB::Array{Float64, 1} = [ c[j] for j in basis ]

	# Find possible feasible solution y to dual problem
	y::Array{Float64, 1} = *(transpose(inv(AB)), cB)

	# If y is feasible for the dual problem, then we are in case 1
	# If y^{T}A_{j} <= c_{j} for j in basis, then y is feasible for the dual problem
	yfeas = [ (*(transpose(y), A[1:m, j])[1] - c[j]) for j in setdiff(1:n, basis) ]
	# println("y^{T}A_{j} - c_{j} for j in basis = ", yfeas)
	sindex = findfirst(x -> x > 0, yfeas)
	sindex > 0 && ((case1 = false); s = setdiff(1:n, basis)[sindex])

	###	Case 1	###
	# If we are in case 1, then x is an optimal solution to original LPP
	if case1 == true
		return x, true, (c'x)[1]

	else 	###	Case 2	###
		# If we are in case 2, we must check whether we are in subcase 1 or subcase 2
		# Set the m-dimensional vector t
		t = *(inv(AB), A[1:m, s])
		# println("t = ", t)

		# if t[i] <= 0 for i in basis then we are in subcase 1
		findfirst(x -> x > 0, t) > 0 && (subcase1 = false)

		###	Subcase 1	###
		# If we are in subcase 1, then there is no optimal solution
		subcase1 == false || return x, false, -Inf

		###	Subcase 2	###
		# If we are in subcase 2, we find a new basic feasible solution to the LPP with the following:
		# We first find the largest lambda such that xnew(lambda) is feasible
		lambda = minimum([ x[basis[i]]/t[i] for i in find(x -> x > 0 , t)])

		# Set the new basic feasible solution xnew
		xnew[s] = lambda
		for h in 1:length(basis)
			xnew[basis[h]] = x[basis[h]] - lambda*t[h]
		end

		# println(*(A, xnew) - b)
		# println("The cost of x is ", *(transpose(c), x))
		# println("The cost of xnew is ", *(transpose(c), xnew))
		return xnew, false, (c'x)[1]
	end
end

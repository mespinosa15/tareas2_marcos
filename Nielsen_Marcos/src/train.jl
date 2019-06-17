"""
    sgd!(network, traindata; epochs, mini_batch_size, η, test_data)

Train the neural network using mini-batch stochastic
gradient descent.  The ``training_data`` is a list of tuples
``(x, y)`` representing the training inputs and the desired
outputs.  The other non-optional parameters are
self-explanatory.  If ``test_data`` is provided then the
network will be evaluated against the test data after each
epoch, and partial progress printed out.  This is useful for
tracking progress, but slows things down substantially.
"""
function sgd!(net::Network{L,Lm1}, train_data;
              epochs::Int = 30, mini_batch_size::Int = 10,
              η::Real = 3.0, test_data = nothing) where {L,Lm1}
    @assert epochs > 0
    @assert mini_batch_size > 0

    a = ntuple(l -> zeros(size(net,l)), L)
    z = ntuple(l -> zeros(size(net,l+1)), Lm1)
    ∇b = ntuple(l -> zeros(size(net,l+1)), Lm1)
    ∇w = ntuple(l -> zeros(size(net,l+1), size(net,l)), Lm1)
    δ = ntuple(l -> zeros(size(net,l+1)), Lm1)

    for j = 1:epochs
        shuffle!(train_data)
        for k = mini_batch_size : mini_batch_size : length(train_data)
            mini_batch = train_data[k - mini_batch_size + 1 : k]
            @assert length(mini_batch) == mini_batch_size
            update_mini_batch!(net, a, z, ∇b, ∇w, δ, mini_batch, η)
        end
        if test_data ≠ nothing
            res = evaluate(net, test_data)
            println("Epoch $j: $res / $(length(test_data))")
        else
            println("Epoch $j complete")
        end
    end
end

"""
    update_mini_batch!(network, mini_batch, η)

Update the network's weights and biases by applying
gradient descent using backpropagation to a single mini batch.
The ``mini_batch`` is a list of tuples `(x, y)`, and `η`
is the learning rate.
"""
function update_mini_batch!(net::Network{L,Lm1},
                            a::NTuple{L,Vector{Float64}},
                            z::NTuple{Lm1,Vector{Float64}},
                            ∇b::NTuple{Lm1,Vector{Float64}},
                            ∇w::NTuple{Lm1,Matrix{Float64}},
                            δ::NTuple{Lm1,Vector{Float64}},
                            mini_batch, η::Real) where {L,Lm1}
    @assert 0 < η < Inf
    check_sizes(net, a, z, ∇b, ∇w)
    for v in ∇b v .= 0 end
    for v in ∇w v .= 0 end
    for (input, output) in mini_batch
        backprop!(a, z, ∇b, ∇w, δ, net, input, output)
    end
    for l = 1:L-1
        net.b[l] .-= η .* ∇b[l] ./ length(mini_batch)
        net.w[l] .-= η .* ∇w[l] ./ length(mini_batch)
    end
end

"""
    backprop!(a, z, ∇b, ∇w, δ, network, input, output)

Adds the gradients of the cost component for the example (input, output)
to ∇b and ∇w.
"""
function backprop!(a::NTuple{L,Vector{Float64}},
                   z::NTuple{Lm1,Vector{Float64}},
                   ∇b::NTuple{Lm1,Vector{Float64}},
                   ∇w::NTuple{Lm1,Matrix{Float64}},
                   δ::NTuple{Lm1,Vector{Float64}},
                   net::Network{L,Lm1},
                   input::Vector{<:Real},
                   output::Vector{<:Real}) where {L,Lm1}
    check_sizes(net, a, z, ∇b, ∇w)
    @assert length.(δ) == length.(net.b)
    a[1] .= input
    feedforward!(a, z, net)
    δ[L-1] .= cost_prime.(a[L], output) .* relu_prime.(z[L-1])
    ∇b[L-1] .+= δ[L-1]
    ∇w[L-1] .+= δ[L-1] .* a[L-1]'
    for l = L-2 : -1 : 1
        mul!(δ[l], net.w[l+1]', δ[l+1])
        δ[l] .*= relu_prime.(z[l])
        ∇b[l] .+= δ[l]
        ∇w[l] .+= δ[l] .* a[l]'
    end
end

function check_sizes(net::Network{L,Lm1},
                     a::NTuple{L,Vector{Float64}},
                     z::NTuple{Lm1,Vector{Float64}},
                     ∇b::NTuple{Lm1,Vector{Float64}},
                     ∇w::NTuple{Lm1,Matrix{Float64}}) where {L,Lm1}
    check_sizes(net, a, z)
    @assert length.(∇b) == length.(net.b) == size(net, 2:L)
    @assert size.(∇w) == size.(net.w) == (zip(size(net, 2:L), size(net, 1:L-1))...,)
end

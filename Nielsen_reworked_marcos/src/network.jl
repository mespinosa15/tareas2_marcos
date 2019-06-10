struct Network{L,Lm1}
    sizes::NTuple{L,Int}    # number of neurons in each layer
    b::NTuple{Lm1,Vector{Float64}}    # biases
    w::NTuple{Lm1,Matrix{Float64}}    # weights
    function Network(sizes::NTuple{L,Int}) where {L}
        @assert L > 0 && all(sizes .> 0)
        Lm1 = L - 1
        b = ntuple(l -> randn(sizes[l+1]), Lm1)
        w = ntuple(l -> randn(sizes[l+1], sizes[l]), Lm1)
        new{L,Lm1}(sizes, b, w)
    end
end
Network(sizes::Int...) = Network(sizes)

depth(::Network{L}) where {L} = L
inlen(net::Network) = first(size(net))
outlen(net::Network) = last(size(net))
Base.size(net::Network) = net.sizes
Base.size(net::Network, l) = size(net)[l] # TODO: I don't know if this is needed

"""
    feedforward!(a, z, network)
    feedforward!(a, network)

Writes the value of each neuron in-place into `a`, taking
the input from `a[1]`. In particular the network output is
written at `a[end]`. If `z` is provided, writes the
"weighted inputs" into `z`.
"""
function feedforward!(a::NTuple{L, Vector{Float64}},
                      z::NTuple{Lm1, Vector{Float64}},
                      net::Network{L,Lm1}) where {L,Lm1}
    check_sizes(net, a, z)
    for l = 1:L-1
        mul!(z[l], net.w[l], a[l])
        z[l] .+= net.b[l]
        a[l+1] .= sigmoid.(z[l],i)
    end
end
feedforward!(a::NTuple{L,Vector{Float64}}, net::Network{L}) where {L} =
    feedforward!(a, a[2:end], net)

"""
    feedforward(input, network)

Returns output vector of `network` on the `input` given.
"""
function feedforward(input::Vector{Float64}, net::Network{L}) where {L}
    @assert length(input) == inlen(net)
    a = ntuple(l -> zeros(size(net, l)), L)
    a[1] .= input
    feedforward!(a, net)
    return a[end]
end

"""
    evaluate(network, test_data)

Return the number of test inputs for which the neural
network outputs the correct result. Note that the neural
network's output is assumed to be the index of whichever
neuron in the final layer has the highest activation.
"""
function evaluate(net::Network, test_data)
    a = ntuple(l -> zeros(size(net, l)), depth(net))
    count = 0
    for (input, output) in test_data
        a[1] .= input
        feedforward!(a, net)
        @assert length(output) == outlen(net) == length(a[end])
        if findmax(a[end])[2] == findmax(output)[2]
            count += 1
        end
    end
    return count
end

function check_sizes(net::Network{L,Lm1},
                     a::NTuple{L,Vector{Float64}},
                     z::NTuple{Lm1,Vector{Float64}}) where {L,Lm1}
    @assert length.(a) == size(net)
    @assert length.(z) == size(net,2:L)
end

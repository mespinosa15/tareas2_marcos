#= Isolate these functions here so I can change them easily =#

"ReLU function"

relu(z::Number) =  max(z,0.02z)

relu_prime(z::Number ) = max(z,0.02z) / z

"derivative of cost w.r.t. to activation of a neuron in last layer"
cost_prime(a::Number, y::Number) = a - y # L2 cost

#= Isolate these functions here so I can change them easily =#

"sigmoid function"
sigmoid(z::Number,i::Number) = ((i == 0) ? ( one(z) / (one(z) + exp(-z)) ) : ( max(z,0.01*z) ) )
"derivative of sigmoid function"
sigmoid_prime(z::Number,i::Number) = ((i == 0) ? ( exp(-z) / (1 + exp(-z))^2) : ( max(z,0.01*z) / z ))

"derivative of cost w.r.t. to activation of a neuron in last layer"
cost_prime(a::Number, y::Number) = a - y # L2 cost

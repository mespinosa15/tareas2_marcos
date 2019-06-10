function load_minst(datadir::String)
    train_x = readdlm(datadir * "/train_in.txt")
    train_y = readdlm(datadir * "/train_out.txt")
    train_y = [[i == y ? 1 : 0 for i = 0:9] for y in train_y]
    @assert size(train_x, 1) == size(train_y, 1)
    traindata = [(train_x[i,:], train_y[i]) for i = 1:size(train_x, 1)]

    tests_x = readdlm(datadir * "/tests_in.txt");
    tests_y = readdlm(datadir * "/tests_out.txt");
    tests_y = [[i == y ? 1 : 0 for i = 0:9] for y in tests_y]
    @assert size(tests_x, 1) == size(tests_y, 1)
    testsdata = [(tests_x[i,:], tests_y[i]) for i = 1:size(tests_x, 1)]

    valid_x = readdlm(datadir * "/valid_in.txt");
    valid_y = readdlm(datadir * "/valid_out.txt");
    valid_y = [[i == y ? 1 : 0 for i = 0:9] for y in valid_y]
    @assert size(valid_x, 1) == size(valid_y, 1)
    validdata = [(valid_x[i,:], valid_y[i]) for i = 1:size(valid_x, 1)]

    return traindata, testsdata, validdata
end

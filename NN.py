def construct_perceptron(weights, bias):
    """Returns a perceptron function using the given paramers."""
    def perceptron(input):
        # Complete (a line or two)
        sums = 0
        for i in range(len(weights)):
            sums += weights[i] * input[i]
        # Note: we are masking the built-in input function but that is
        # fine since this only happens in the scope of this function and the
        # built-in input is not needed here.
        sums += bias
        if sums >= 0:
            return 1
        return 0
    
    return perceptron # this line is fine


# test cases

weights = [2, -4]
bias = 0
perceptron = construct_perceptron(weights, bias)

print(perceptron([1, 1]))
print(perceptron([2, 1]))
print(perceptron([3, 1]))
print(perceptron([-1, -1]))


def accuracy(classifier, inputs, expected_outputs):
    lists = []
    for i in inputs:
        lists.append(classifier(i))
    r_count = 0
    for i in range(len(lists)):
        if lists[i] == expected_outputs[i]:
            r_count += 1
    return r_count / len(lists)


#test cases

perceptron = construct_perceptron([-1, 3], 2)
inputs = [[1, -1], [2, 1], [3, 1], [-1, -1]]
targets = [0, 1, 1, 0]

print(accuracy(perceptron, inputs, targets))

perceptron = construct_perceptron([1, -3], 2)
inputs = [[1, -1], [2, 1], [3, 1], [-1, -1]]
targets = [0, 1, 1, 0]

print(accuracy(perceptron, inputs, targets))


def learn_perceptron_parameters(weights, bias, training_examples, learning_rate, max_epochs):
    weights = weights
    bias = bias
    for _ in range(max_epochs):
        for i in training_examples:
            if (bias + (weights[0] * i[0][0]) + (weights[1] * i[0][1])) < 0:
                y = 0
            else:
                y = 1
            t = i[1]

            if y != t:
                weights[0] = weights[0] + learning_rate * (i[0][0])*(t - y)
                weights[1] = weights[1] + learning_rate * (i[0][1])*(t - y)
                bias = bias + learning_rate * (t-y)
    return weights, bias

#test cases

weights = [2, -4]
bias = 0
learning_rate = 0.5
examples = [
  ((0, 0), 0),
  ((0, 1), 0),
  ((1, 0), 0),
  ((1, 1), 1),
  ]
max_epochs = 50

weights, bias = learn_perceptron_parameters(weights, bias, examples, learning_rate, max_epochs)
print(f"Weights: {weights}")
print(f"Bias: {bias}\n")

perceptron = construct_perceptron(weights, bias)

print(perceptron((0,0)))
print(perceptron((0,1)))
print(perceptron((1,0)))
print(perceptron((1,1)))
print(perceptron((2,2)))
print(perceptron((-3,-3)))
print(perceptron((3,-1)))


weights = [2, -4]
bias = 0
learning_rate = 0.5
examples = [
  ((0, 0), 0),
  ((0, 1), 1),
  ((1, 0), 1),
  ((1, 1), 0),
  ]
max_epochs = 50

weights, bias = learn_perceptron_parameters(weights, bias, examples, learning_rate, max_epochs)
print(f"Weights: {weights}")
print(f"Bias: {bias}\n")
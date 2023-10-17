from math import sqrt

def euclidean_distance(v1, v2):
  summation = 0.0
  for i in range(len(v1)):

    summation += (v2[i] - v1[i])**2

  return sqrt(summation)
 
def majority_element(labels):
    amount = {}
    for i in labels:
        if i in amount:
            amount[i] += 1
        else:
            amount[i] = 1
    maxi = 0
    most = None
    for l, f in amount.items():
        if f > maxi:
            maxi = f
            most = l
    return most


# test cases

print(majority_element([0, 0, 0, 0, 0, 1, 1, 1]))
print(majority_element("ababc") in "ab")

print(majority_element([0, 0, 0, 0, 0, 1, 1, 1]))
print(majority_element("ababc") in "ab")


def knn_predict(input, examples, distance, combine, k):
    holding = []
    for (number, operator) in examples:
        eucl = distance(input, number)
        holding.append((eucl, operator))
    holding = sorted(holding)

    using = holding[:k]
    for example in holding[k:]:
        if using[-1][0] != example[0]:
            break
        else:
            using.append(example)

    holder = []
    for _, i in using:
        holder.append(i)

    return combine(holder)

#tests for knn predict

examples = [
    ([2], '-'),
    ([3], '-'),
    ([5], '+'),
    ([8], '+'),
    ([9], '+'),
]

distance = euclidean_distance
combine = majority_element

for k in range(1, 6, 2):
    print("k =", k)
    print("x", "prediction")
    for x in range(0,10):
        print(x, knn_predict([x], examples, distance, combine, k))
    print()


# using knn for predicting numeric values

examples = [
    ([1], 5),
    ([2], -1),
    ([5], 1),
    ([7], 4),
    ([9], 8),
]

def average(values):
    return sum(values) / len(values)

distance = euclidean_distance
combine = average

for k in range(1, 6, 2):
    print("k =", k)
    print("x", "prediction")
    for x in range(0,10):
        print("{} {:4.2f}".format(x, knn_predict([x], examples, distance, combine, k)))
    print()
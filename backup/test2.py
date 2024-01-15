from itertools import product

l1 = [1, 2, 3, 4]
l2 = ['a', 'b', 'c']
l3 = [3.14, 2.71]

l = list(product(l1, l2, l3))
print(l)

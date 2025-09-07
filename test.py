def last_Digit_Factorial(n):    if n == 0 or n == 1:        return 1\n    if n >= 5:\n        return 0\n    factorial = 1\n    for i in range(2, n + 1):\n        factorial *= i\n    return factorial % 10



assert max_sum_pair_diff_lessthan_K([3, 5, 10, 15, 17, 12, 9], 7, 4) == 62
assert max_sum_pair_diff_lessthan_K([5, 15, 10, 300], 4, 12) == 25
assert max_sum_pair_diff_lessthan_K([1, 2, 3, 4, 5, 6], 6, 6) == 21

print(max_sum_pair_diff_lessthan_K([3, 5, 10, 15, 17, 12, 9], 7, 4))
print(max_sum_pair_diff_lessthan_K([5, 15, 10, 300], 4, 12))
print(max_sum_pair_diff_lessthan_K([1, 2, 3, 4, 5, 6], 6, 6))
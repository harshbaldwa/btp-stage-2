import numpy as np

def counting_sort(A, digit, radix):
    B = [0]*len(A)
    C = [0]*int(radix)

    for i in range(len(A)):
        digit_of_Ai = int((A[i]/radix**digit)%radix)
        C[digit_of_Ai] += 1
    
    for j in range(1, radix):
        C[j] += C[j-1]

    for m in range(len(A)-1, -1, -1):
        digit_of_Ai = int((A[m]/radix**digit)%radix)
        C[digit_of_Ai] -= 1
        B[C[digit_of_Ai]] = A[m]

    return B
### Dmitry Salimov
### Project Euler solutions file
### https://projecteuler.net/

#%% Clear variable list
def clearall():
    """clear all globals"""
    for uniquevar in [var for var in globals().copy() if var[0] != "_" and var != 'clearall']:
        del globals()[uniquevar]
clearall()

#%% 1) Multiples of 3 and 5
#If we list all the natural numbers below 10 that are multiples of 3 or 5, we get 3, 5, 6 and 9. The sum of these multiples is 23.
#Find the sum of all the multiples of 3 or 5 below 1000.
def multiples(n):
    num = 1
    while num < n:
        if num % 3 == 0 or num % 5 == 0: 
            yield num
        num += 1

sum(multiples(1000))

sum([x for x in range(1000) if x % 3 == 0 or x % 5 == 0])

#%% 2) Even Fibonacci numbers
#Each new term in the Fibonacci sequence is generated by adding the previous two terms. By starting with 1 and 2, the first 10 terms will be:
#1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...
#By considering the terms in the Fibonacci sequence whose values do not exceed four million, find the sum of the even-valued terms.

def fib(n):
    if n > 2:
        return fib(n-2) + fib(n-1)
    else:
        return n
        
def fibonacci(n):
    num = 1
    while fib(num) < n:
        yield fib(num)
        num += 1

sum([x for x in fibonacci(4000000) if x % 2 == 0])

#%% 3) Largest prime factor
#The prime factors of 13195 are 5, 7, 13 and 29.
#What is the largest prime factor of the number 600851475143 ?

def primes_naive(n):
    if n < 2: return []
    sieve = [True] * n
    for i in range(3,int(n**0.5)+1,2):
        if sieve[i]:
            sieve[i*i::2*i]=[False]*((n-i*i-1)//(2*i)+1)
    return [2] + [i for i in range(3,n,2) if sieve[i]]

def primes(n):
    sieve = [True] * (n//2)
    for i in range(3,int(n**0.5)+1,2):
        if sieve[i//2]:
            sieve[i*i//2::i] = [False] * ((n-i*i-1)//(2*i)+1)
    return [2] + [2*i+1 for i in range(1, n//2) if sieve[i]]
      
def prod(x):
    product = 1
    for i in x:
        product *= i
    return product
    
def prime_factors(n, m):
    nums = [x for x in primes(m) if n % x == 0]
    if prod(nums) == n:
        return nums
    else:
        return []

max(prime_factors(600851475143, 10000))  

#%% 4) Largest palindrome product
#A palindromic number reads the same both ways. The largest palindrome made from the product of two 2-digit numbers is 9009 = 91 × 99.
#Find the largest palindrome made from the product of two 3-digit numbers.
def palindrome(n):
    if str(n) == str(n)[::-1]:
        return True
    else:
        return False

def palindromes(n):
    nums = []
    for i in range(10**(n-1), 10**n):
        for j in range(i, 10**n):
            if i*j not in nums and palindrome(i*j):
                nums.append(i*j)
    return nums

max(palindromes(3))

#%% 5) Smallest multiple
#2520 is the smallest number that can be divided by each of the numbers from 1 to 10 without any remainder.
#What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?
def decompose(lst):
    if lst and lst != [1]:
        ls = [x for x in lst if x != 1]
        elem = ls[0]
        l = [x // elem if x % elem == 0 else x for x in ls]
        return [elem] + decompose(l)
    else:
        return []

def prod(x):
    product = 1
    for i in x:
        product *= i
    return product

prod(decompose(list(range(1, 21))))

#%% 6) Sum square difference
#The sum of the squares of the first ten natural numbers is 1^2 + 2^2 + ... + 10^2 = 385
#The square of the sum of the first ten natural numbers is (1 + 2 + ... + 10)^2 = 55^2 = 3025
#Hence the difference between the sum of the squares of the first ten natural numbers and the square of the sum is 3025 − 385 = 2640.
#Find the difference between the sum of the squares of the first one hundred natural numbers and the square of the sum.
nums = list(range(1, 101))
sum_squares = sum([x**2 for x in nums])
square_sum = sum(nums)**2
square_sum - sum_squares

#%% 7) 10001st prime
#By listing the first six prime numbers: 2, 3, 5, 7, 11, and 13, we can see that the 6th prime is 13.
#What is the 10 001st prime number?
def prime(n):
    primes = [2, 3, 5, 7]
    if n < 5:
        yield primes[n-1]
    else:
        candidate = 11
        while len(primes) <= n:
            limit = candidate ** 0.5
            for p in primes:
                if p > limit:
                    primes.append(candidate)
                    break
                if candidate % p == 0:
                    break
            candidate += 2
        yield primes[n-1]
        
list(prime(10001))[0]

#%% 8) Largest product in a series
#The four adjacent digits in the 1000-digit number that have the greatest product are 9 × 9 × 8 × 9 = 5832.
#Find the thirteen adjacent digits in the 1000-digit number that have the greatest product. What is the value of this product?
num = """
73167176531330624919225119674426574742355349194934
96983520312774506326239578318016984801869478851843
85861560789112949495459501737958331952853208805511
12540698747158523863050715693290963295227443043557
66896648950445244523161731856403098711121722383113
62229893423380308135336276614282806444486645238749
30358907296290491560440772390713810515859307960866
70172427121883998797908792274921901699720888093776
65727333001053367881220235421809751254540594752243
52584907711670556013604839586446706324415722155397
53697817977846174064955149290862569321978468622482
83972241375657056057490261407972968652414535100474
82166370484403199890008895243450658541227588666881
16427171479924442928230863465674813919123162824586
17866458359124566529476545682848912883142607690042
24219022671055626321111109370544217506941658960408
07198403850962455444362981230987879927244284909188
84580156166097919133875499200524063689912560717606
05886116467109405077541002256983155200055935729725
71636269561882670428252483600823257530420752963450
""".replace('\n', '')

def prod(x):
    product = 1
    for i in x:
        product *= int(i)
    return product

l = 13
nums = [num[x:x+l] for x in range(len(num)-l+1)]
prods = [prod(x) for x in nums]

max(prods)

#%% 9) Special Pythagorean triplet
#A Pythagorean triplet is a set of three natural numbers, a < b < c, for which a^2 + b^2 = c^2.
#For example, 3^2 + 4^2 = 9 + 16 = 25 = 5^2.
#There exists exactly one Pythagorean triplet for which a + b + c = 1000.
#Find the product abc.
def pythagorean_naive(n):
    for a in range(0+1, n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                if a+b+c == n:
                    if a**2 + b**2 == c**2:
                        return [a, b, c]
                    else:
                        continue
                else:
                    continue
    return []

def pythagorean(n):
    for a in range(1, n//3):
        for b in range(a+1, n//2):
            if 2*a*b-2*n*(a+b)+n**2 == 0:
                return [a, b, int((a**2+b**2)**0.5)]
            else:
                continue

def prod(x):
    product = 1
    for i in x:
        product *= i
    return product
    
prod(pythagorean(1000))

#%% Summation of primes
#The sum of the primes below 10 is 2 + 3 + 5 + 7 = 17.
#Find the sum of all the primes below two million.
def primes(n):
    sieve = [True] * (n//2)
    for i in range(3,int(n**0.5)+1,2):
        if sieve[i//2]:
            sieve[i*i//2::i] = [False] * ((n-i*i-1)//(2*i)+1)
    return [2] + [2*i+1 for i in range(1, n//2) if sieve[i]]

sum(primes(2000000))

#%% 11) Largest product in a grid
#In the 20×20 grid below, four numbers along a diagonal line have been marked in red.
#The product of these numbers is 26 × 63 × 78 × 14 = 1788696.
#What is the greatest product of four adjacent numbers in the same direction (up, down, left, right, or diagonally) in the 20×20 grid?
grid = """
08 02 22 97 38 15 00 40 00 75 04 05 07 78 52 12 50 77 91 08
49 49 99 40 17 81 18 57 60 87 17 40 98 43 69 48 04 56 62 00
81 49 31 73 55 79 14 29 93 71 40 67 53 88 30 03 49 13 36 65
52 70 95 23 04 60 11 42 69 24 68 56 01 32 56 71 37 02 36 91
22 31 16 71 51 67 63 89 41 92 36 54 22 40 40 28 66 33 13 80
24 47 32 60 99 03 45 02 44 75 33 53 78 36 84 20 35 17 12 50
32 98 81 28 64 23 67 10 26 38 40 67 59 54 70 66 18 38 64 70
67 26 20 68 02 62 12 20 95 63 94 39 63 08 40 91 66 49 94 21
24 55 58 05 66 73 99 26 97 17 78 78 96 83 14 88 34 89 63 72
21 36 23 09 75 00 76 44 20 45 35 14 00 61 33 97 34 31 33 95
78 17 53 28 22 75 31 67 15 94 03 80 04 62 16 14 09 53 56 92
16 39 05 42 96 35 31 47 55 58 88 24 00 17 54 24 36 29 85 57
86 56 00 48 35 71 89 07 05 44 44 37 44 60 21 58 51 54 17 58
19 80 81 68 05 94 47 69 28 73 92 13 86 52 17 77 04 89 55 40
04 52 08 83 97 35 99 16 07 97 57 32 16 26 26 79 33 27 98 66
88 36 68 87 57 62 20 72 03 46 33 67 46 55 12 32 63 93 53 69
04 42 16 73 38 25 39 11 24 94 72 18 08 46 29 32 40 62 76 36
20 69 36 41 72 30 23 88 34 62 99 69 82 67 59 85 74 04 36 16
20 73 35 29 78 31 90 01 74 31 49 71 48 86 81 16 23 57 05 54
01 70 54 71 83 51 54 69 16 92 33 48 61 43 52 01 89 19 67 48
""".replace('\n', ' ').strip()

n = 20
matrix = [grid.split(' ')[n*i:n*(i+1)] for i in range(n)]
matrix_trans = list(zip(*matrix))
matrix_diag = [[matrix[n-1-q][p-q] for q in range(min(p, n-1), max(0, p-n+1)-1, -1)] for p in range(n+n-1)]
matrix_rev = [list(reversed(x)) for x in matrix]
matrix_rev_diag = list(reversed([[matrix_rev[n-1-q][p-q] for q in range(min(p, n-1), max(0, p-n+1)-1, -1)] for p in range(n+n-1)]))

def prod(x):
    product = 1
    for i in x:
        product *= int(i)
    return product
    
def prod_n(l, n=4):
    if len(l) >= n:
        return [prod(l[x:x+n]) for x in range(len(l)-n+1)]
    else:
        return []

prods = [prod_n(x) for x in matrix + matrix_trans + matrix_diag + matrix_rev_diag]
max([max(x) for x in prods if x])
    
#%% 12) Highly divisible triangular number
#The sequence of triangle numbers is generated by adding the natural numbers. So the 7th triangle number would be 1 + 2 + 3 + 4 + 5 + 6 + 7 = 28. The first ten terms would be:
#1, 3, 6, 10, 15, 21, 28, 36, 45, 55, ...
#Let us list the factors of the first seven triangle numbers:
# 1: 1
# 3: 1,3
# 6: 1,2,3,6
#10: 1,2,5,10
#15: 1,3,5,15
#21: 1,3,7,21
#28: 1,2,4,7,14,28
#We can see that 28 is the first triangle number to have over five divisors.
#What is the value of the first triangle number to have over five hundred divisors?
import functools  
def factors(n):    
    return set(functools.reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))    

def ntriangle(n):
    i = 2 
    while len(factors(sum(range(i)))) <= n:
        i += 1
    return sum(range(i))

#%% 13) Large sum
#Work out the first ten digits of the sum of the following one-hundred 50-digit numbers.

nums = """
37107287533902102798797998220837590246510135740250
...
53503534226472524250874054075591789781264330331690
""".replace('\n', ' ').strip().split(' ')

str(sum([int(x) for x in nums]))[:10]

##%% 14) Longest Collatz sequence
#The following iterative sequence is defined for the set of positive integers:
#n → n/2 (n is even)
#n → 3n + 1 (n is odd)
#Using the rule above and starting with 13, we generate the following sequence: 13 → 40 → 20 → 10 → 5 → 16 → 8 → 4 → 2 → 1
#It can be seen that this sequence (starting at 13 and finishing at 1) contains 10 terms. Although it has not been proved yet (Collatz Problem), it is thought that all starting numbers finish at 1.
#Which starting number, under one million, produces the longest chain?
#NOTE: Once the chain starts the terms are allowed to go above one million.
def collatz(n):
    nums = []
    while n > 1:
        nums.append(n)
        n=(n//2,n*3+1)[int(n%2)]
    return nums+[1]
    
def collatz_count(n, d = {1: 1}):
    stack = []
    while n not in d:
        stack.append(n)
        n = n * 3 + 1 if n & 1 else n // 2
    c = d[n]
    while stack:
        c += 1
        d[stack.pop()] = c
    return c

counts = [(x, collatz_count(x)) for x in range(1, 10**6)]
sorted(counts, key=lambda tup: tup[1], reverse = True)[0]

#%% 15) Lattice paths
#Starting in the top left corner of a 2×2 grid, and only being able to move to the right and down, there are exactly 6 routes to the bottom right corner.
#How many such routes are there through a 20×20 grid?
import math
def npermutations(n, k):
    return math.factorial(n)//(math.factorial(k)*math.factorial(n-k))
    
npermutations(40, 20)

#%% 16) Power digit sum
#2^15 = 32768 and the sum of its digits is 3 + 2 + 7 + 6 + 8 = 26.
#What is the sum of the digits of the number 2^1000?
sum([int(x) for x in str(2**1000)])

#%% 17) Number letter counts
#If the numbers 1 to 5 are written out in words: one, two, three, four, five, then there are 3 + 3 + 5 + 4 + 4 = 19 letters used in total.
#If all the numbers from 1 to 1000 (one thousand) inclusive were written out in words, how many letters would be used?
#NOTE: Do not count spaces or hyphens. For example, 342 (three hundred and forty-two) contains 23 letters and 115 (one hundred and fifteen) contains 20 letters. The use of "and" when writing out numbers is in compliance with British usage.
digits = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
teens = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
decades = ['', 'ten', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']

def spell(n):
    if n < 10:
        return digits[n]
    elif n < 20:
        return teens[n-10]
    elif n < 100:
        return decades[n//10] + "-" + spell(n%10)
    elif n < 1000:
        if n%100 == 0:
            return spell(n//100) + " hundred"
        else:
            return spell(n//100) + " hundred and " + spell(n%100)
    elif n < 1000000:
        return spell(n//1000) + " thousand " + spell(n%1000)
        
words = [spell(x) for x in list(range(1, 1001))]
len(''.join(words).replace('-', '').replace(' ', ''))

#%% 18) Maximum path sum I
#By starting at the top of the triangle below and moving to adjacent numbers on the row below, the maximum total from top to bottom is 23.
#3
#7 4
#2 4 6
#8 5 9 3
#That is, 3 + 7 + 4 + 9 = 23.
#Find the maximum total from top to bottom of the triangle below:
#NOTE: As there are only 16384 routes, it is possible to solve this problem by trying every route. However, Problem 67, is the same challenge with a triangle containing one-hundred rows; it cannot be solved by brute force, and requires a clever method! ;o)
triangle = """
75
95 64
17 47 82
18 35 87 10
20 04 82 47 65
19 01 23 75 03 34
88 02 77 73 07 63 67
99 65 04 28 06 16 70 92
41 41 26 56 83 40 80 70 33
41 48 72 33 47 32 37 16 94 29
53 71 44 65 25 43 91 52 97 51 14
70 11 33 28 77 73 17 78 39 68 17 57
91 71 52 38 17 14 91 43 58 50 27 29 48
63 66 04 68 89 53 67 30 73 16 69 87 40 31
04 62 98 27 23 09 70 98 73 93 38 53 60 04 23
""".split("\n")
triangle = [x.split(" ") for x in triangle if x]

def max_path(x):
    y = [[int(i) for i in j] for j in x]
    while len(y) > 1:
        y[-2] = [y[-2][i] + max(y[-1][i], y[-1][i+1]) for i in range(len(y[-2]))]
        y.pop()
    return y[0][0]        

max_path(triangle)

#%% 19) Counting Sundays
#You are given the following information, but you may prefer to do some research for yourself.
#1 Jan 1900 was a Monday.
#Thirty days has September,
#April, June and November.
#All the rest have thirty-one,
#Saving February alone,
#Which has twenty-eight, rain or shine.
#And on leap years, twenty-nine.
#A leap year occurs on any year evenly divisible by 4, but not on a century unless it is divisible by 400.
#How many Sundays fell on the first of the month during the twentieth century (1 Jan 1901 to 31 Dec 2000)?
import datetime
limits = [datetime.date(1901, 1, 1), datetime.date(2000, 12, 31)]
dates = [limits[0] + datetime.timedelta(days=x) for x in range(0, (limits[1] - limits[0]).days + 1)]
first = [x for x in dates if x.day == 1]
sundays = [x for x in first if x.weekday() == 6]
len(sundays)

#%% 20)Factorial digit sum
#n! means n × (n − 1) × ... × 3 × 2 × 1
#For example, 10! = 10 × 9 × ... × 3 × 2 × 1 = 3628800,
#and the sum of the digits in the number 10! is 3 + 6 + 2 + 8 + 8 + 0 + 0 = 27.
#Find the sum of the digits in the number 100!
import math
sum([int(x) for x in str(math.factorial(100))])

#%% 21) Amicable numbers
#Let d(n) be defined as the sum of proper divisors of n (numbers less than n which divide evenly into n).
#If d(a) = b and d(b) = a, where a ≠ b, then a and b are an amicable pair and each of a and b are called amicable numbers.
#For example, the proper divisors of 220 are 1, 2, 4, 5, 10, 11, 20, 22, 44, 55 and 110; therefore d(220) = 284. The proper divisors of 284 are 1, 2, 4, 71 and 142; so d(284) = 220.
#Evaluate the sum of all the amicable numbers under 10000.

#import math
#def fac(n):
#    step = lambda x: 1 + x*4 - x//2*2
#    maxq = math.floor(math.sqrt(n))
#    d = 1
#    q = n % 2 == 0 and 2 or 3 
#    while q <= maxq and n % q != 0:
#        q = step(d)
#        d += 1
#    return q <= maxq and [q] + fac(n//q) or [n]

def div(n):
    return [x for x in range(1, n) if n%x==0]
    
def amic(m, n):
    return True if sum(div(m))==n and sum(div(n))==m else False
    
A = [(x,sum(div(x)), sum(div(sum(div(x))))) for x in range(1, 10000)]
B = list(set([(min(x), max(x)) for x in A if x[0]==x[2] and x[0]!=x[1]]))
sum([sum(x) for x in B])

#%% 22) Names scores
#Using names.txt (right click and 'Save Link/Target As...'), a 46K text file containing over five-thousand first names, begin by sorting it into alphabetical order. Then working out the alphabetical value for each name, multiply this value by its alphabetical position in the list to obtain a name score.
#For example, when the list is sorted into alphabetical order, COLIN, which is worth 3 + 15 + 12 + 9 + 14 = 53, is the 938th name in the list. So, COLIN would obtain a score of 938 × 53 = 49714.
#What is the total of all the name scores in the file?
import string
with open("D:\p022_names.txt", mode='r') as doc:
    names = doc.read().replace('"', '').split(',')
    
def score(name):
    return sum([string.ascii_uppercase.index(l)+1 for l in name])

names_sorted = sorted(names)
names_scores = [score(x)*(names_sorted.index(x)+1) for x in names_sorted]
sum(names_scores)

#%% 23) Non-abundant sums
#A perfect number is a number for which the sum of its proper divisors is exactly equal to the number. For example, the sum of the proper divisors of 28 would be 1 + 2 + 4 + 7 + 14 = 28, which means that 28 is a perfect number.
#A number n is called deficient if the sum of its proper divisors is less than n and it is called abundant if this sum exceeds n.
#As 12 is the smallest abundant number, 1 + 2 + 3 + 4 + 6 = 16, the smallest number that can be written as the sum of two abundant numbers is 24. By mathematical analysis, it can be shown that all integers greater than 28123 can be written as the sum of two abundant numbers. However, this upper limit cannot be reduced any further by analysis even though it is known that the greatest number that cannot be expressed as the sum of two abundant numbers is less than this limit.
#Find the sum of all the positive integers which cannot be written as the sum of two abundant numbers.
import itertools

def div(n):
    if n == 1: return [1]
    return [x for x in range(1, n) if n%x==0]

nums_all = [(x, div(x)) for x in range(1, 28124)]
nums_abundant = [x for x, y in nums_all if sum(y) > x]
nums_sum2 = list(set([sum(x) for x in itertools.combinations(nums_abundant, 2) if sum(x) < 28124] + [x*2 for x in nums_abundant]))
nums_result = [x for x in range(1, 28124) if x not in nums_sum2]
sum(nums_result)

#%% 24) Lexicographic permutations
#A permutation is an ordered arrangement of objects. For example, 3124 is one possible permutation of the digits 1, 2, 3 and 4. If all of the permutations are listed numerically or alphabetically, we call it lexicographic order. The lexicographic permutations of 0, 1 and 2 are:
#012   021   102   120   201   210
#What is the millionth lexicographic permutation of the digits 0, 1, 2, 3, 4, 5, 6, 7, 8 and 9?
import math

def sub(n, i):
    if n > 0 and i > 0:
        c = 0
        while n > math.factorial(i):
            n -= math.factorial(i)
            c += 1
        return [(n, c*math.factorial(i), c)] + sub(n, i-1)
    else:
        return [(n, 0, 0)]
        
pos = [z for x, y, z in sub(10**6, 9)]

def shift(l):
    string = list(range(10))
    result = []
    for i in l:
        result.append(string[i])
        string.pop(i)
    return "".join([str(x) for x in result])
        
shift(pos)
    
#%%




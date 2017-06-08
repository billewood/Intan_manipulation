# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:12:21 2017

@author: billewood
"""

def base10toBaseN(num, base_n):
#    for i = range(digits):
    i = 0
    baseN_num = list()
    while num != 0:
        remainder = num % base_n
        num = num / base_n
        baseN_num.append(remainder)
        i += 1
    s = reduce(lambda x,y: x+str(y), baseN_num, '')
    return int(s)
    
def baseNtoBase10(num, base_n):
    num_list = [int(x) for x in str(num)]
#    num_list = list(reversed(num_list))
    converted = [x * (base_n ** ind) for ind, x in enumerate(num_list)]
    converted[0] = num_list[0]
    return sum(converted)
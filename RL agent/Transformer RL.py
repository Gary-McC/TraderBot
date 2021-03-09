import numpy as np

test="this is a test"
string1="abcd efg"
string2="gfed cba"

def Anagram(string1, string2):
    
    string1=string1+'?'
    string2=string2+'?'
    
    string1=''.join(sorted(string1)).split('?')[-1]
    string2=''.join(sorted(string2)).split('?')[-1]
    return string1==string2


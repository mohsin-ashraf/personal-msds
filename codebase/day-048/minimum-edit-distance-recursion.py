


string1 = "Ahsan"
string2 = "Mohsin"

def editDistance(str1,str2,m,n):
    """
        str1,str2: Strings to match or for which we have to find minimum edit distance.
        m: length of str1
        n: length of str2
    """
    # If the str1 is empty then we need to add "n" charactors from str2 to a null string to make the str1 equal to str2
    if m == 0:
        return n
    
    # If str2 is empty then we need to add "m" charactors from str2 to a null string to make the str2 equal to str1
    if n == 0:
        return m

    if str1[m-1] == str2[n-1]:
        return editDistance(str1,str2,m-1,n-1)

    # If last characters are not same, consider all three 
    # operations on last character of first string, recursively 
    # compute minimum cost for all three operations and take 
    # minimum of three values. 
    # DELETE,INSERT, UPDATE

    return 1 + min(editDistance(str1,str2,m,n-1),editDistance(str1,str2,m-1,n),editDistance(str1,str2,m-1,n-1)) # Special case for substitutoin Levenstein distance



print ("Minimum distance for converting {} to {} is {}".format(string1,string2,editDistance(string1.lower(),string2.lower(),len(string1),len(string2))))

    

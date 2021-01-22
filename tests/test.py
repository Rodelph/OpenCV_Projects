def func(*args, **kwargs):
    # *args means for however many arguments you take in, it will catch them all
 
    for arg in args:
        print(arg)
     
 
l = [11,3,4,5,"tuts"]
 
print(func(l,k = 1))
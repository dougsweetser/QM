
# coding: utf-8

# In[4]:


a=((1,  1,  1,   1), # matrix A #
     (2,  4,  8,  16),
     (3,  9, 27,  81),
     (4, 16, 64, 256))
 
b=((  4  , -3  ,  4/3.,  -1/4. ), # matrix B #
     (-13/3., 19/4., -7/3.,  11/24.),
     (  3/2., -2.  ,  7/6.,  -1/4. ),
     ( -1/6.,  1/4., -1/6.,   1/24.))
 
 
 
def MatrixMul( mtx_a, mtx_b):
    tpos_b = zip( *mtx_b)
    rtn = [[ sum( ea*eb for ea,eb in zip(a,b)) for b in tpos_b] for a in mtx_a]
    return rtn
 
 
v = MatrixMul( a, b )
 
print('v = (')
for r in v:
    print('[') 
    for val in r:
        print('%8.2f '%val) 
    print(']')
print(')')
 
 
u = MatrixMul(b,a)
 
print('u = ')
for r in u:
    print('[') 
    for val in r:
        print('%8.2f '%val) 
    print(']')
print(')')


# In[8]:


m = [[1], [2], [3], [4]]
print("m: ", m)
tm = zip( *m)

for t in tm:
    print("t: ", t)


# In[10]:


# 3 rows, 1 column
a=[[1], [2], [3]]
# 1 row, 3 columns
b=[[4, 5, 6]]

# ab is: 3.1 * 1.3 = 3x3 matrix, 9 elements
ab = MatrixMul(a, b)

# ba is: 1.3 * 3.1 = 1x1 matrix, 1 element
ba = MatrixMul(b, a)

print("ab: ", ab)
print("ba: ", ba)


import tensorflow as tf
# This piece of code contains following information
# 1. how to create a tensor with normal distribution and given shape
# 2. Multiply a tensor with a scalar value
# 3. How to concat/stack two or more tensors with same second dimension
# 4. Create a copy of a tensor using identity
# 5. How to update a specific row in 2D tensor


tensor1 = tf.constant([[5., 2.],
                       [1., 2.],
                       [4., 3.]])
print("tensor 1 : ", tensor1)
old_N, D = tensor1.shape
print(old_N, " x ", D)
new_N = 2 + old_N
new_tensor = tf.random.normal(shape=[new_N, D])
#print(new_tensor)
new_tensor = tf.scalar_mul(0.01, new_tensor)
print("tensor new with new_N : ")
print(new_tensor)
# new_weight = old_weight.new(new_N, D).normal_().mul_(std)
#new_tensor[:old_N] = tf.identity(tensor1)
t = new_tensor[old_N:]
#print("tensor new after slicing : ")
final = tf.concat([tf.identity(tensor1), t], 0)
print("Final Tensor : ")
print(final)
p = tf.random.normal(shape=[1, D])
print(p)
final = tf.tensor_scatter_nd_update(final, [[2]], p)
print(final)

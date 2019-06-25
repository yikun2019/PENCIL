import pickle
import numpy as np

def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

# r is noise rate

r = 0.1

count = 0

# add symmetric noise

a = unpickle('./cifar-10-batches-py/data_batch_1')
for i in range(10000):
    if np.random.random()< r:
        a['labels'][i] = np.random.randint(0,10)
        count += 1
with open('./cifar-10-batches-py/data_batch_1','wb') as file:
    pickle.dump(a,file)

a = unpickle('./cifar-10-batches-py/data_batch_2')
for i in range(10000):
    if np.random.random() <  r:
        a['labels'][i] = np.random.randint(0, 10)
        count += 1
with open('./cifar-10-batches-py/data_batch_2', 'wb') as file:
    pickle.dump(a, file)

a = unpickle('./cifar-10-batches-py/data_batch_3')
for i in range(10000):
    if np.random.random() <  r:
        a['labels'][i] = np.random.randint(0, 10)
        count += 1
with open('./cifar-10-batches-py/data_batch_3', 'wb') as file:
    pickle.dump(a, file)

a = unpickle('./cifar-10-batches-py/data_batch_4')
for i in range(10000):
    if np.random.random() <  r:
        a['labels'][i] = np.random.randint(0, 10)
        count += 1
with open('./cifar-10-batches-py/data_batch_4', 'wb') as file:
    pickle.dump(a, file)

a = unpickle('./cifar-10-batches-py/data_batch_5')
for i in range(5000):
    if np.random.random() <  r:
        a['labels'][i] = np.random.randint(0, 10)
        count += 1
with open('./cifar-10-batches-py/data_batch_5', 'wb') as file:
    pickle.dump(a, file)


print(count)
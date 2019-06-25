import pickle
import numpy as np

def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

# r is noise rate

r = 0.1

# add asymmetric noise

a = unpickle('./cifar-10-batches-py/data_batch_1')
for i in range(10000):
    if np.random.random() < r:
        if a['labels'][i] == 9:
            a['labels'][i] = 1
        elif a['labels'][i] == 2:
            a['labels'][i] = 0
        elif a['labels'][i] == 4:
            a['labels'][i] = 7
        elif a['labels'][i] == 3:
            a['labels'][i] = 5
        elif a['labels'][i] == 5:
            a['labels'][i] = 3
with open('./cifar-10-batches-py/data_batch_1','wb') as file:
    pickle.dump(a,file)

a = unpickle('./cifar-10-batches-py/data_batch_2')
for i in range(10000):
    if np.random.random() < r:
        if a['labels'][i] == 9:
            a['labels'][i] = 1
        elif a['labels'][i] == 2:
            a['labels'][i] = 0
        elif a['labels'][i] == 4:
            a['labels'][i] = 7
        elif a['labels'][i] == 3:
            a['labels'][i] = 5
        elif a['labels'][i] == 5:
            a['labels'][i] = 3
with open('./cifar-10-batches-py/data_batch_2','wb') as file:
    pickle.dump(a,file)

a = unpickle('./cifar-10-batches-py/data_batch_3')
for i in range(10000):
    if np.random.random() < r:
        if a['labels'][i] == 9:
            a['labels'][i] = 1
        elif a['labels'][i] == 2:
            a['labels'][i] = 0
        elif a['labels'][i] == 4:
            a['labels'][i] = 7
        elif a['labels'][i] == 3:
            a['labels'][i] = 5
        elif a['labels'][i] == 5:
            a['labels'][i] = 3
with open('./cifar-10-batches-py/data_batch_3','wb') as file:
    pickle.dump(a,file)

a = unpickle('./cifar-10-batches-py/data_batch_4')
for i in range(10000):
    if np.random.random() < r:
        if a['labels'][i] == 9:
            a['labels'][i] = 1
        elif a['labels'][i] == 2:
            a['labels'][i] = 0
        elif a['labels'][i] == 4:
            a['labels'][i] = 7
        elif a['labels'][i] == 3:
            a['labels'][i] = 5
        elif a['labels'][i] == 5:
            a['labels'][i] = 3
with open('./cifar-10-batches-py/data_batch_4','wb') as file:
    pickle.dump(a,file)

a = unpickle('./cifar-10-batches-py/data_batch_5')
for i in range(5000):
    if np.random.random() < r:
        if a['labels'][i] == 9:
            a['labels'][i] = 1
        elif a['labels'][i] == 2:
            a['labels'][i] = 0
        elif a['labels'][i] == 4:
            a['labels'][i] = 7
        elif a['labels'][i] == 3:
            a['labels'][i] = 5
        elif a['labels'][i] == 5:
            a['labels'][i] = 3
with open('./cifar-10-batches-py/data_batch_5','wb') as file:
    pickle.dump(a,file)



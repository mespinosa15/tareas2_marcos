# Python 2
# execute in data directory of Nielsen's repo

import numpy as np
import cPickle
import gzip

with gzip.open('mnist.pkl.gz', 'rb') as f:
	training_data, validation_data, test_data = cPickle.load(f)

np.savetxt('train_in.txt', training_data[0])
np.savetxt('train_out.txt', training_data[1])
np.savetxt('tests_in.txt', test_data[0])
np.savetxt('tests_out.txt', test_data[1])
np.savetxt('valid_in.txt', validation_data[0])
np.savetxt('valid_out.txt', validation_data[1])

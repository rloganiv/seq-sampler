import numpy as np
import pandas as pd
import process
import sampler

# Parameters
k = 0.0
beta = 0.0

# Load sequence data
df = pd.read_csv('./data/traj-noloop-all-Melb.csv')
seqs, vocab = process.extract_seqs(df, value_id='poiID', sort_id='startTime',
                                   group_id='userID')
alpha, gamma = process.transition_matrix(seqs, vocab, k)

# Initialize sampler
sampler = sampler.MCSampler(alpha, gamma, beta)

# Draw test samples
print 'Generating test sequences'
for _ in xrange(10):
    print sampler.gen_sequence()

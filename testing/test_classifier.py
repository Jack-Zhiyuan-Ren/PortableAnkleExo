# prepare data for estimation, subtract one cond from other
# offset handles height/weight which we don't want to subtract from each other
import pickle
import numpy as np
#from sklearn import linear_model

def label_pairwise_mat(mat, offset=2):
    pairwise_len = 0
    for i in range(0,mat.shape[0]-1):
        pairwise_len += (i+1)
    pairwise_data = np.zeros((pairwise_len, mat.shape[1]))
    pairwise_conds = np.zeros((pairwise_len, 2),dtype=int)
    counter = 0
    for i in range(mat.shape[0]-1): # cond 1 index
        for j in range(i+1, mat.shape[0]): # cond 2 index
            pairwise_data[counter,:offset] = mat[i,:offset]
            pairwise_data[counter,offset:] = mat[i,offset:] - mat[j,offset:]
            pairwise_conds[counter,:] = [i,j]
            counter += 1
    return pairwise_data, pairwise_conds

def compute_ordering_from_pairs(pair_ests, pairwise_conds, order_size, confidence=[-1.0]):
    if confidence[0] == -1.0:
        confidence = np.ones(order_size)
    order_scores = np.zeros(order_size)
    num_pairs = pairwise_conds.shape[0]
    for i in range(num_pairs):
        cond1, cond2 = pairwise_conds[i,:]
        val = 1.0*confidence[i]**2
        if pair_ests[i] == 0: # label 0 means i < j which makes i better
            order_scores[cond1] -= val
            order_scores[cond2] += val
        else: # any other label measn j < i which makes j better
            order_scores[cond1] += val
            order_scores[cond2] -= val
            
    est_ordering = np.argsort(order_scores)
    overlap = order_size - len(set(order_scores))
    #print("Estimates:", order_scores, est_ordering)
    return est_ordering, overlap, np.sort(order_scores)+100.0

def classifyModel(modelname, pw1_data):
    loaded_model = pickle.load(open(modelname, 'rb'))# load model
    #print(pw1_data.shape)
    pw1 = loaded_model.predict(pw1_data)
    pw1_prob = loaded_model.predict_proba(pw1_data)
    pw1_prob = np.amax(pw1_prob,axis=1) # take max across classes
    return pw1,pw1_prob

order_size = 8 # now 6, was 8
modelname = 'lr_model.pkl'
test_data_name = 'subj_data.npy'
test_data = np.load(test_data_name)
test_data = test_data.T
print(test_data.shape)
pw1_data, pairwise_conds = label_pairwise_mat(test_data) # input is [ordersize, 66]
pw1, pw1_prob = classifyModel(modelname, pw1_data) # load classifier 
est_ordering, overlap, order_scores = compute_ordering_from_pairs(pw1, pairwise_conds, order_size, pw1_prob)
unorder_args = np.argsort(est_ordering)
unorder_scores = order_scores[unorder_args] # scores for the conditions in order they were tested in CMA
print(unorder_scores)
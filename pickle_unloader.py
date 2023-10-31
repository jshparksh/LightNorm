import pickle

with open('./_pickles/0_fwd_act.pkl', 'rb') as fr:
    input_pkl = pickle.load(fr)
print(input_pkl)
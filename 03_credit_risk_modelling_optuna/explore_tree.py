import lightgbm as lgbm
import matplotlib.pyplot as plt
import pickle

with open('lgbm_model.pickle', 'rb') as file:
    lgbm_model = pickle.load(file)

lgbm.plot_tree(lgbm_model, tree_index = 3)
plt.show()
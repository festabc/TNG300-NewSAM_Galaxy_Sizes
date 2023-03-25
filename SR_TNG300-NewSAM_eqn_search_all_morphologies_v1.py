import time
import numpy as np
import pandas as pd

import galsim #install with conda install -c conda_forge galsim

import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib.cm as cm
import matplotlib.colors as norm
from matplotlib.gridspec import SubplotSpec
import seaborn as sns

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline #This allows one to build different steps together
from sklearn.preprocessing import StandardScaler, RobustScaler

from tqdm import tqdm 

from pysr import PySRRegressor

import os
os.mkdir('TNG300-SAM_images/SR_run4_all_morph_eqn_search')

#### Physical Model Equation search in all galaxies (that is, disks + ellipticals at once)

# Normalized dataset: all masses divided by halo mass (Mvir)

df_normalized_35 = pd.read_csv('TNG300-SAM_images/v1_TNG300-SAM_cleanup_normalize_dataset/TNG300-NewSAM_Normalized_Dataset_fromv1_wo_mstar9_nonphys_and_diskgals_w_smallfdsik.csv')

df_all = df_normalized_35.loc[:, :]

X_all = df_all.drop(columns=['GalpropNormHalfRadius', 'BulgeMstar_ratio', 'GalpropNormMdisk', 
                                        'DiskMstar_ratio'])

y_all = df_all.loc[:,'GalpropNormHalfRadius']


df_all_sample = df_all.sample(n = 10000, random_state = 2023) 

# choose a subset of randomly sampled data, 
# fix random seed here because we don't want to have the random seed effect when we repeat the eqn search with more iterations
# choose a subset of 10,000 randomly sampled galaxies because Symbolic Regression takes a max of 10K entries


# choose only the 7 most important features from feature ranking in notebook v13,
# in order to reduce the time to run SR modelling

X_all_imp = df_all_sample.loc[:, ['HalopropSpin', 'GalpropNormMHII', 'GalpropNormSigmaBulge',
                                  'GalpropNormMstar_merge','GalpropNormMstar', 'HalopropC_nfw',
                                  'GalpropNormMbulge']]

### The most important features in TNG300, without $log_{10}M_{star}$<9.0, ($f_{disk}$ <0.0205 & $\frac{M_{bulge}}{M_{star}}$ < 0.4),  (using randomly chosen 100,000 out of 207,000 galaxies):

# 1 HalopropSpin 0.652298282114003
# 2 GalpropNormMHII 0.7619957833330049
# 3 GalpropNormSigmaBulge 0.8439434958639686
# 4 GalpropNormMstar_merge 0.8864413423026826
# 5 GalpropNormMstar 0.923864263978027
# 6 HalopropC_nfw 0.9436933580615018
# 7 GalpropNormMbulge 0.954425272690842
# 8 GalpropOutflowRate_Mass 0.9553352815152003
# 9 GalpropTmerger_major 0.9560816642231309
# 10 GalpropTmerger 0.9566102956279537
# 11 GalpropNormVdisk 0.9571773024099443


y_all_imp = df_all_sample.loc[:, 'GalpropNormHalfRadius']

# choose the Symbolic Regression model; choose the mathematical operations allowed
model_all_imp = PySRRegressor(
    
    niterations=10000,
    
    unary_operators=[ "square"], #, "exp", "cube", "log10_abs", "log1p"] ,#   "inv(x) = 1/x"
#         "inv(x) = 1/x",  # Custom operator (julia syntax)
         
    
    binary_operators=["+", "-", "*", "pow", "/"], #"mylogfunc(x)=log(1-(x))" ],


    constraints={
        "pow": (4, 1),
        "/": (-1, 4),
#         "log1p": 4,
    },
    
    # extra_sympy_mappings={'mylogfunc': lambda x: log1p(x)},
    
    nested_constraints={
        "pow": {"pow": 0}, #, "exp": 0},
        "square": {"square": 0} #, "cube": 0, "exp": 0},
#         "cube": {"square": 0, "cube": 0, "exp": 0},
#         "exp": {"square": 0, "cube": 0, "exp": 0},
#         "log1p": {"pow": 0, "exp": 0},
    },
    
    maxsize=30,
    multithreading=False,
    model_selection="best", # Result is mix of simplicity+accuracys
    loss="loss(x, y) = (x - y)^2"  # Custom loss function (julia syntax)

)

start_time = time.time()

model_all_imp.fit(X_all_imp, np.array(y_all_imp))

elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the SymbolicRegression fitting for all morphologies: {elapsed_time:.3f} seconds")

# Run1 with n_iter=1,000, n_galaxies = 10,000, random_state fixed, unary and binary operators included 
# (unary_operators=[ "square"],  binary_operators=["+", "-", "*", "pow", "/"], constraints={"pow": (4, 1), "/":(-1, 4)}
#  nested_constraints={ "pow": {"pow": 0}, "square": {"square": 0}   
# Run2 with n_iter=5,000, everything else same as in Run 1
# Run3 with n_iter=10,000, everything else same as in Run 1
# Run4 with n_iter=15,000, everything else same as in Run 1

model_all_imp.equations_
all_eqns = model_all_imp.equations_
all_eqns.to_csv('TNG300-SAM_images/SR_run4_all_morph_eqn_search/run4_SR_All_morphologies_equations_n_iter_15000')

all_pred = model_all_imp.predict(X_all_imp)
all_pred = pd.DataFrame(all_pred)
all_pred.to_csv('TNG300-SAM_images/SR_run4_all_morph_eqn_search/run4_SR_All_morphologies_Predicted_sizes_n_iter_15000')


print(model_all_imp.sympy())


r2_score_all=r2_score(y_all_imp, model_all_imp.predict(X_all_imp))


with open('TNG300-SAM_images/SR_run4_all_morph_eqn_search/run4_SR_All_morphologies_bestequation_n_iter_15000.txt', 'w', encoding='utf-8') as txt_save:
    txt_save.write('The most important features used by SR to find the best equation are:')
    txt_save.write(str(X_all_imp.columns.to_list()))
    txt_save.write('\n \n')
    txt_save.write('Run4: The best equation with n_iter 15,000 is:')
    txt_save.write(str(model_all_imp.sympy()))
    txt_save.write('\n \n')
    txt_save.write('R2 score=' + '{:.2f}'.format(r2_score_all))
    txt_save.write('\n \n')
    txt_save.write('Elapsed time to compute the All Morphologies SR =' + '{:.3f}'.format(elapsed_time) + 'seconds')




plt.scatter(y_all_imp, model_all_imp.predict(X_all_imp),
            c = df_all_sample['GalpropNormMbulge']/df_all_sample['GalpropNormMstar'],  cmap='Spectral_r',
            s=10, marker='.', alpha=0.7, label = r'$\frac{M_{bulge}}{M_{star}}$', vmin=0.0, vmax=0.5) #,label= label,
plt.axis([0.0,0.2, 0.0,0.2])
plt.plot([0.0, 0.3], [0.0, 0.3], color = 'black', linewidth = 2)
plt.text(0.02, 0.17, 'R2 score=' + '{:.2f}'.format(r2_score_all), size=12)
plt.text(0.02, 0.15, 'eqn=' + '{}'.format(model_all_imp.sympy()), size=10)
plt.title('Predicted vs True Dimensionless Galaxy Size with SR \n All Morphologies')
plt.xlabel('True Dimensionless Galaxy Size')
plt.ylabel('Predicted Dimensionless Galaxy Size ')
plt.legend(loc='lower right' , shadow=True)
plt.colorbar()
plt.savefig('TNG300-SAM_images/SR_run4_all_morph_eqn_search/run4_SR_All_morphologies_predicted_vs_true_size_n_iter_15000.jpeg', dpi=500)
plt.show()



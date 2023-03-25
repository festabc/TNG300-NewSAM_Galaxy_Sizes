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
os.mkdir('TNG300-SAM_images/SR_df_1_run5_pure_disks_eqn_search')

#### Physical Model Equation search in df_1, pure disk galaxies only

# Normalized dataset: all masses divided by halo mass (Mvir)

df_1 = pd.read_csv('TNG300-SAM_images/v4_TNG300-SAM_wo_DISKgals_w_smallfdisk/v4_df_1_Pure_Disks_Normalized_as_defined_in_TNG300notebook_v4')


df_disks = df_1.loc[:, :]

df_disks_sample = df_disks.sample(n = 10000, random_state = 2023) 

# choose a subset of randomly sampled data, 
# fix random seed here because we don't want to have the random seed effect when we repeat the eqn search with more iterations
# choose a subset of 10,000 randomly sampled galaxies because Symbolic Regression takes a max of 10K entries


# choose only the 7 most important features from feature ranking in notebook v13,
# in order to reduce the time to run SR modelling

X_disks_imp = df_disks_sample.loc[:, ['HalopropSpin', 'HalopropC_nfw', 'GalpropSfrave20myr',
                                      'GalpropNormMHII', 'GalpropNormMstar',
                                      'GalpropNormMHI', 'GalpropNormSigmaBulge']]

### The most important features for Group 1 (df_1) - pure disk galaxies - in TNG300, without $log_{10}M_{star}$<9.0, ($f_{disk}$ <0.0205 & $\frac{M_{bulge}}{M_{star}}$ < 0.4):

# 1 HalopropSpin 0.9316749236373595
# 2 HalopropC_nfw 0.9843207557546408
# 3 GalpropSfrave20myr 0.9910818401670922
# 4 GalpropNormMHII 0.992994979871547
# 5 GalpropNormMstar 0.9940427165531078
# 6 GalpropNormMHI 0.9943019214926601
# 7 GalpropNormSigmaBulge 0.994460482770335
# 8 GalpropSfr 0.9944317021260475
# 9 HalopropMaccdot_reaccreate 0.9943410958906713
# 10 GalpropSfrave1gyr 0.9942507285793947
# 11 GalpropNormMstar_merge 0.9942589651516193
# 12 GalpropZstar 0.9943024431030104
# 13 HalopropZhot 0.9941144906849857
# 14 GalpropNormVdisk 0.9941315737811508
# 15 GalpropMaccdot_radio 0.9940293979071809
# 16 HalopropMcooldot 0.9940791054443174
# 17 HalopropNormMstar_diffuse 0.9939457432393111


y_disks_imp = df_disks_sample.loc[:, 'GalpropNormHalfRadius']

# # choose the Symbolic Regression model; choose the mathematical operations allowed
# model_disks_imp = PySRRegressor(
    
#     niterations=5000,
    
#     unary_operators=[ "square"], #, "exp", "cube", "log10_abs", "log1p"] ,#   "inv(x) = 1/x"
# #         "inv(x) = 1/x",  # Custom operator (julia syntax)
         
    
#     binary_operators=["+", "-", "*", "pow", "/"], #"mylogfunc(x)=log(1-(x))" ],


#     constraints={
#         "pow": (4, 1),
#         "/": (-1, 4),
# #         "log1p": 4,
#     },
    
#     # extra_sympy_mappings={'mylogfunc': lambda x: log1p(x)},
    
#     nested_constraints={
#         "pow": {"pow": 0}, #, "exp": 0},
#         "square": {"square": 0} #, "cube": 0, "exp": 0},
# #         "cube": {"square": 0, "cube": 0, "exp": 0},
# #         "exp": {"square": 0, "cube": 0, "exp": 0},
# #         "log1p": {"pow": 0, "exp": 0},
#     },
    
#     maxsize=30,
#     multithreading=False,
#     model_selection="best", # Result is mix of simplicity+accuracys
#     loss="loss(x, y) = (x - y)^2"  # Custom loss function (julia syntax)

# )

# choose the Symbolic Regression model; choose the mathematical operations allowed
model_disks_imp = PySRRegressor(
    
    niterations=1000,
    
    unary_operators=["exp", "square", "cube", "log10_abs", "log1p"] ,#   "inv(x) = 1/x"
#         "inv(x) = 1/x",  # Custom operator (julia syntax)
         
    
    binary_operators=["+", "*", "pow", "/"], #"mylogfunc(x)=log(1-(x))" ],


        constraints={
        "pow": (4, 1),
        "/": (-1, 4),
        "log1p": 4,
    },
    
    # extra_sympy_mappings={'mylogfunc': lambda x: log1p(x)},
    
    nested_constraints={
        "pow": {"pow": 0, "exp": 0},
        "square": {"square": 0, "cube": 0, "exp": 0},
        "cube": {"square": 0, "cube": 0, "exp": 0},
        "exp": {"square": 0, "cube": 0, "exp": 0},
        "log1p": {"pow": 0, "exp": 0},
    },
    
    maxsize=30,
    multithreading=False,
    model_selection="best", # Result is mix of simplicity+accuracys
    loss="loss(x, y) = (x - y)^2"  # Custom loss function (julia syntax)

)

start_time = time.time()

model_disks_imp.fit(X_disks_imp, np.array(y_disks_imp))

elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the SymbolicRegression fitting for df_1, pure disk morphologies: {elapsed_time:.3f} seconds")

# Run1 with n_iter=1,000, n_galaxies = 10,000, random_state fixed, unary and binary operators included 
# (unary_operators=[ "square"],  binary_operators=["+", "-", "*", "pow", "/"], constraints={"pow": (4, 1), "/":(-1, 4)}
#  nested_constraints={ "pow": {"pow": 0}, "square": {"square": 0}   
# Run2 with n_iter=5,000, everything else same as in Run 1
# Run3 same as Run2, but w only 2 imp features
# Run4 with n_iter=1,000, only 2 imp features, and more operators in pySR model selection
# Run5, same as Run4 but with 7 imp features

# Run3 with n_iter=10,000, everything else same as in Run 1
# Run4 with n_iter=15,000, everything else same as in Run 1

model_disks_imp.equations_
disks_eqns = model_disks_imp.equations_
disks_eqns.to_csv('TNG300-SAM_images/SR_df_1_run5_pure_disks_eqn_search/run5_SR_pure_disks_equations_n_iter_1000')

disks_pred = model_disks_imp.predict(X_disks_imp)
disks_pred = pd.DataFrame(disks_pred)
disks_pred.to_csv('TNG300-SAM_images/SR_df_1_run5_pure_disks_eqn_search/run5_SR_pure_disks_Predicted_sizes_n_iter_1000')


print(model_disks_imp.sympy())


r2_score_disks=r2_score(y_disks_imp, model_disks_imp.predict(X_disks_imp))


with open('TNG300-SAM_images/SR_df_1_run5_pure_disks_eqn_search/run5_SR_pure_disks_bestequation_n_iter_1000.txt', 'w', encoding='utf-8') as txt_save:
    txt_save.write('The most important features used by SR to find the best equation are:')
    txt_save.write(str(X_disks_imp.columns.to_list()))
    txt_save.write('\n \n')
    txt_save.write('Run5: The best equation with n_iter 1,000 is:')
    txt_save.write(str(model_disks_imp.sympy()))
    txt_save.write('\n \n')
    txt_save.write('R2 score=' + '{:.2f}'.format(r2_score_disks))
    txt_save.write('\n \n')
    txt_save.write('Elapsed time to compute the Pure Disk Morphologies SR =' + '{:.3f}'.format(elapsed_time) + 'seconds')




plt.scatter(y_disks_imp, model_disks_imp.predict(X_disks_imp),
            c = df_disks_sample['GalpropNormMbulge']/df_disks_sample['GalpropNormMstar'],  cmap='Spectral_r',
            s=10, marker='.', alpha=0.7, label = r'$\frac{M_{bulge}}{M_{star}}$', vmin=0.0, vmax=0.5) #,label= label,
plt.axis([0.0,0.2, 0.0,0.2])
plt.plot([0.0, 0.3], [0.0, 0.3], color = 'black', linewidth = 2)
plt.text(0.02, 0.17, 'R2 score=' + '{:.2f}'.format(r2_score_disks), size=12)
plt.text(0.02, 0.15, 'eqn=' + '{}'.format(model_disks_imp.sympy()), size=10)
plt.title('Predicted vs True Dimensionless Galaxy Size with SR \n' + r'Group 1: $\frac{M_{bulge}}{M_{star}}$<=0.10  Pure Disks')
plt.xlabel('True Dimensionless Galaxy Size')
plt.ylabel('Predicted Dimensionless Galaxy Size ')
plt.legend(loc='lower right' , shadow=True)
plt.colorbar()
plt.savefig('TNG300-SAM_images/SR_df_1_run5_pure_disks_eqn_search/run5_SR_pure_disks_predicted_vs_true_size_n_iter_1000.jpeg', dpi=500)
plt.show()



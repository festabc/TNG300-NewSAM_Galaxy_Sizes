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
os.mkdir('TNG300-SAM_images/SR_Residuals_df_14_run2_disks_eqn_search')

#### Physical Model Equation search for Residuals in df_14, disk galaxies (with spin_effective variable introduced)

# Normalized dataset: all masses divided by halo mass (Mvir)


df_14 = pd.read_csv('TNG300-SAM_images/v6_TNG300-SAM_Morphologies_definition/df_14_Normalized_as_defined_in_TNG300notebook_v6')

#### Add a column for the predictions from the SR equation Spin/$NormVdisk^{2}$ and the Residuals of these predictions vs true values (for the Normalized Dataset only)

# Add the SR predicted equation as a column to the df
df_14.loc[:, 'SpinNormVdisk2_eqn'] = df_14.loc[:, 'HalopropSpin']/df_14.loc[:, 'GalpropNormVdisk']**2

# define Residuals
df_14.loc[:, 'Residuals'] = df_14.loc[:, 'GalpropNormHalfRadius'] - df_14.loc[:, 'SpinNormVdisk2_eqn']

# choose a subsample of 10K galaxies since SR doesn't work with a larger dataset
df_disks_sample = df_14.sample(n = 10000, random_state = 2023) 

# choose a subset of randomly sampled data, 
# fix random seed here because we don't want to have the random seed effect when we repeat the eqn search with more iterations
# choose a subset of 10,000 randomly sampled galaxies because Symbolic Regression takes a max of 10K entries


# choose only the 7 most important features from feature ranking in notebook v13,
# in order to reduce the time to run SR modelling

X_disks_imp = df_disks_sample.loc[:, ['GalpropNormSigmaBulge', 'GalpropOutflowRate_Mass', 'GalpropNormMcold',
                                      'GalpropTmerger', 'HalopropC_nfw', 'GalpropNormMstar', 'GalpropNormMbulge',
                                      'GalpropSfr', 'GalpropMstrip', 'HalopropSpin'
                                      ]] 

### The most important features used to predict Residuals of Groups 1-4 (df_14) - all disk galaxies - in TNG300 (see Notebook v11):

#  1 GalpropNormSigmaBulge 0.1506724241589664
#  2 GalpropOutflowRate_Mass 0.31667643276347335
#  3 GalpropNormMcold 0.4402794564186907
#  4 GalpropTmerger 0.5440037604970072
#  5 HalopropC_nfw 0.5943951438514031
#  6 GalpropNormMstar 0.6308015490156137
#  7 GalpropNormMbulge 0.6854078292436087
#  8 GalpropSfr 0.6987805615374164
#  9 GalpropMstrip 0.7057863290505919
#  10 HalopropSpin 0.7090032534211362


y_disks_imp = df_disks_sample.loc[:, 'Residuals']  # we want to predict Residuals

# choose the Symbolic Regression model; choose the mathematical operations allowed
model_disks_imp = PySRRegressor(
    
    niterations=5000,
    
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

#     procs=7
)


start_time = time.time()

model_disks_imp.fit(X_disks_imp, np.array(y_disks_imp))

elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the SymbolicRegression fitting for df_14, all disk morphologies: {elapsed_time:.3f} seconds")

# Run1 with n_iter=1,000, n_galaxies = 10,000, random_state fixed, unary and binary operators included 
# (unary_operators=[ "square"],  binary_operators=["+", "-", "*", "pow", "/"], constraints={"pow": (4, 1), "/":(-1, 4)}
#  nested_constraints={ "pow": {"pow": 0}, "square": {"square": 0}   ,
# ['GalpropNormSigmaBulge', 'GalpropOutflowRate_Mass', 'GalpropNormMcold',
#                                       'GalpropTmerger', 'HalopropC_nfw', 'GalpropNormMstar', 'GalpropNormMbulge',
#                                       'GalpropSfr', 'GalpropMstrip', 'HalopropSpin'
#                                       ] as important features
# Run2 same as Run1 but with n_iter=5,000

model_disks_imp.equations_
disks_eqns = model_disks_imp.equations_
disks_eqns.to_csv('TNG300-SAM_images/SR_Residuals_df_14_run2_disks_eqn_search/run2_SR_all_disks_equations_n_iter_5K')

disks_pred = model_disks_imp.predict(X_disks_imp)
disks_pred = pd.DataFrame(disks_pred)
disks_pred.to_csv('TNG300-SAM_images/SR_Residuals_df_14_run2_disks_eqn_search/run2_SR_all_disks_Predicted_Residuals_n_iter_5K')


print(model_disks_imp.sympy())


r2_score_disks=r2_score(y_disks_imp, model_disks_imp.predict(X_disks_imp))


with open('TNG300-SAM_images/SR_Residuals_df_14_run2_disks_eqn_search/run2_SR_all_disks_bestequation_n_iter_5K.txt', 'w', encoding='utf-8') as txt_save:
    txt_save.write('The most important features used by SR to find the best equation that predicts Residuals are:')
    txt_save.write(str(X_disks_imp.columns.to_list()))
    txt_save.write('\n \n')
    txt_save.write('Run2: The best equation with n_iter 5,000 is:')
    txt_save.write(str(model_disks_imp.sympy()))
    txt_save.write('\n \n')
    txt_save.write('R2 score=' + '{:.2f}'.format(r2_score_disks))
    txt_save.write('\n \n')
    txt_save.write('Elapsed time to compute the All Disk Morphologies SR =' + '{:.3f}'.format(elapsed_time) + 'seconds')
    txt_save.write('\n \n')
    txt_save.write('Run2 with n_iter=5,000, n_galaxies = 10,000, random_state fixed, unary and binary operators included (unary_operators=[ "square"],  binary_operators=["+", "-", "*", "pow", "/"], constraints={"pow": (4, 1), "/":(-1, 4)}; nested_constraints={ "pow": {"pow": 0}, "square": {"square": 0}) \n The definition of Residuals is \n  Residuals = NormHalfRadius - Spin/NormVdisk^2 (SR predicted eqn for sizes) ')



plt.scatter(y_disks_imp, model_disks_imp.predict(X_disks_imp),
            c = df_disks_sample['GalpropNormMbulge']/df_disks_sample['GalpropNormMstar'],  cmap='Spectral_r',
            s=10, marker='.', alpha=0.7, label = r'$\frac{M_{bulge}}{M_{star}}$', vmin=0.0, vmax=0.4) #,label= label,
plt.axis([-0.05,0.05, -0.05,0.05])
plt.plot([-0.05, 0.05], [-0.05, 0.05], color = 'black', linewidth = 2)
plt.text(-0.04, 0.03, r'$R^{2}$ score=' + '{:.2f}'.format(r2_score_disks), size=12)
plt.text(-0.04, 0.022, 'eqn=' + '{}'.format(model_disks_imp.sympy()), size=10)
plt.title('Predicted vs True Dimensionless Galaxy Size Residuals with SR \n' + r'Groups 1-4: $\frac{M_{bulge}}{M_{star}}$<=0.4 All Disks')
plt.xlabel('True Dimensionless Galaxy Size')
plt.ylabel('Predicted Dimensionless Galaxy Size ')
plt.legend(loc='lower right' , shadow=True)
plt.colorbar()
plt.savefig('TNG300-SAM_images/SR_Residuals_df_14_run2_disks_eqn_search/run2_SR_all_disks_predicted_vs_true_Residuals_n_iter_1K.jpeg', dpi=500)
plt.show()



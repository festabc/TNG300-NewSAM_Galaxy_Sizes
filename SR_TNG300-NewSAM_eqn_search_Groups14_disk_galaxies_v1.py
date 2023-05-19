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
os.mkdir('TNG300-SAM_images/SR_df_14_run10_disks_eqn_search')

#### Physical Model Equation search in df_14, disk galaxies (with spin_effective variable introduced)

# Normalized dataset: all masses divided by halo mass (Mvir)

# df_14_sample = pd.read_csv('TNG300-SAM_images/v6_TNG300-SAM_Morphologies_definition/df_14_Normalized_as_defined_in_TNG300notebook_v6_10Ksubsample_seed2023') # this is the 10K dataset obtained by randomly sampling the Groups1-4 dataset. The morphology distribution of this dataset follows that of the original Groups 1-4 dataset. We think that because galaxies in Group 1 and 2 are overrepresented, while galaxies in Groups 3 and 4 are underrepresented, the SR cannot find a fit for that addresses the change in morphology. Therefore, we introduce the 10K dataset below which has an equal number of galaxies from each group included.

df_14_sample =  pd.read_csv('TNG300-SAM_images/v8_TNG300-SAM_df_14_uniform_distribution/df_14_UniformMorph10K_Normalized_as_defined_in_TNG300notebook_v8')

df_disks_sample = df_14_sample.loc[:, :]

# df_disks_sample = df_disks.sample(n = 10000, random_state = 2023) 

# choose a subset of randomly sampled data, 
# fix random seed here because we don't want to have the random seed effect when we repeat the eqn search with more iterations
# choose a subset of 10,000 randomly sampled galaxies because Symbolic Regression takes a max of 10K entries


# choose only the 7 most important features from feature ranking in notebook v13,
# in order to reduce the time to run SR modelling

X_disks_imp = df_disks_sample.loc[:, ['HalopropSpin', 'GalpropNormVdisk',
                                      'BulgeMstar_ratio'
                                      ]] # we add here DiskMstar_ratio because we know that bulge fraction increases within the group and this feature may help SR pick up a term that describes this change

# X_disks_imp = df_disks_sample.loc[:, ['HalopropSpin', 'GalpropNormVdisk']]

### The most important features for Groups 1-4 (df_14) - all disk galaxies - in TNG300 with spin_effective variable introduced (see Notebook v5), without $log_{10}M_{star}$<9.0, ($f_{disk}$ <0.0205 & $\frac{M_{bulge}}{M_{star}}$ < 0.4):

# 	 1 HalopropSpin 0.907514213325463
# 	 2 HalopropC_nfw 0.9593035416599681
# 	 3 GalpropNormSigmaBulge 0.9714015313937755
# 	 4 GalpropTmerger 0.9780648753830881
# 	 5 GalpropNormMcold 0.982638230039134
# 	 6 GalpropNormMbulge 0.9839699069005933
# 	 7 GalpropNormMstar 0.9844076824112746
# 	 8 GalpropNormMHII 0.9850168321831245
# 	 9 GalpropNormMHI 0.985168946491418
# 	 10 GalpropNormMstar_merge 0.9849910915764676


y_disks_imp = df_disks_sample.loc[:, 'GalpropNormHalfRadius']

# choose the Symbolic Regression model; choose the mathematical operations allowed
model_disks_imp = PySRRegressor(
    
    niterations=1000,
    
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
# ['HalopropSpin', 'HalopropC_nfw', 'GalpropNormMstar', 'GalpropNormMcold'] as important features
# Run2 with n_iter=10,000, same operators as in Run1 
# Run3 with n_iter=1,000, same operators as in Run1, but features are changed to include GalpropNormVdisk instead of its highly correlated version of HalopropC_nfw, and GalpropTmerger is removed completely since it did not show up in any of the previous equations
# Run4 same as Run3, but features used are only Spin, NormVidsk and MdiskMstar_ratio
# Run5 same as Run4, but with n_iter=5K
# Run6, with n_iter=1K, but here we use the dataset with a uniform distribution of morphologies
# Run7, same as Run6 but with n_iter=5K
# Run8, same as Run6 but with NormSigmaBulge added as a feature
# Run9, same as Run8 but with BulgeMstar_ratio instead of DiskMstar_ratio
# Run10, same as Run9 but with NormSigmaBulge removed

model_disks_imp.equations_
disks_eqns = model_disks_imp.equations_
disks_eqns.to_csv('TNG300-SAM_images/SR_df_14_run10_disks_eqn_search/run10_SR_all_disks_equations_n_iter_1K')

disks_pred = model_disks_imp.predict(X_disks_imp)
disks_pred = pd.DataFrame(disks_pred)
disks_pred.to_csv('TNG300-SAM_images/SR_df_14_run10_disks_eqn_search/run10_SR_all_disks_Predicted_sizes_n_iter_1K')


print(model_disks_imp.sympy())


r2_score_disks=r2_score(y_disks_imp, model_disks_imp.predict(X_disks_imp))


with open('TNG300-SAM_images/SR_df_14_run10_disks_eqn_search/run10_SR_all_disks_bestequation_n_iter_1K.txt', 'w', encoding='utf-8') as txt_save:
    txt_save.write('The most important features used by SR to find the best equation are:')
    txt_save.write(str(X_disks_imp.columns.to_list()))
    txt_save.write('\n \n')
    txt_save.write('Run10: The best equation with n_iter 1,000 is:')
    txt_save.write(str(model_disks_imp.sympy()))
    txt_save.write('\n \n')
    txt_save.write('R2 score=' + '{:.2f}'.format(r2_score_disks))
    txt_save.write('\n \n')
    txt_save.write('Elapsed time to compute the All Disk Morphologies SR =' + '{:.3f}'.format(elapsed_time) + 'seconds')
    txt_save.write('\n \n')
    txt_save.write('Run10 with n_iter=1,000, n_galaxies = 10,000, random_state fixed, unary and binary operators included (unary_operators=[ "square"],  binary_operators=["+", "-", "*", "pow", "/"], constraints={"pow": (4, 1), "/":(-1, 4)}; nested_constraints={ "pow": {"pow": 0}, "square": {"square": 0}) \n  In Runs1-5 we used the 10K dataset obtained by randomly sampling the Groups1-4 dataset. The morphology distribution of this dataset follows that of the original Groups 1-4 dataset. We think that because galaxies in Group 1 and 2 are overrepresented, while galaxies in Groups 3 and 4 are underrepresented, the SR cannot find a fit for that addresses the change in morphology. Therefore, we introduce the 10K dataset below which has an equal number of galaxies from each group included. In Run6 and afterwards we use this uniform Groups 1-4 dataset. Run8 is same as Run6 but with NormSigmaBulge added as a feature. Run9, same as Run8 but with BulgeMstar_ratio instead of DiskMstar_ratio. Run10, same as Run9 but with NormSigmaBulge removed')



plt.scatter(y_disks_imp, model_disks_imp.predict(X_disks_imp),
            c = df_disks_sample['GalpropNormMbulge']/df_disks_sample['GalpropNormMstar'],  cmap='Spectral_r',
            s=10, marker='.', alpha=0.7, label = r'$\frac{M_{bulge}}{M_{star}}$', vmin=0.0, vmax=0.4) #,label= label,
plt.axis([0.0,0.2, 0.0,0.2])
plt.plot([0.0, 0.3], [0.0, 0.3], color = 'black', linewidth = 2)
plt.text(0.02, 0.17, 'R2 score=' + '{:.2f}'.format(r2_score_disks), size=12)
plt.text(0.02, 0.15, 'eqn=' + '{}'.format(model_disks_imp.sympy()), size=10)
plt.title('Predicted vs True Dimensionless Galaxy Size with SR \n' + r'Uniform Groups 1-4: $\frac{M_{bulge}}{M_{star}}$<=0.4 All Disks')
plt.xlabel('True Dimensionless Galaxy Size')
plt.ylabel('Predicted Dimensionless Galaxy Size ')
plt.legend(loc='lower right' , shadow=True)
plt.colorbar()
plt.savefig('TNG300-SAM_images/SR_df_14_run10_disks_eqn_search/run10_SR_all_disks_predicted_vs_true_size_n_iter_1K.jpeg', dpi=500)
plt.show()



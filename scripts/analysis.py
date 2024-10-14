#%%

import pandas as pd
from ceva.extreme_value_models import BlockMaximaModel, ThresholdExcessModel, PointProcesModel
# %%

# GEV
#####

data = pd.read_csv("../data/portpirie.csv")

extremes = data["SeaLevel"]
num_years = data["Year"].max() - data["Year"].min() + 1

# Coles (2001) p.59 Examples 3.4.1.
model = BlockMaximaModel(extremes=extremes, num_years=num_years)
model.fit()
model.model_evaluation_plot(alpha=0.5)

#%%
#model.return_level([10,100])["upper"]
#return_level = model.return_level(100)
#{key: val.round(2) for (key, val) in return_level.items()}
#%%
# GPD
#####

data = pd.read_csv("../data/rain.csv", parse_dates=[0])
years = [date.year for date in data["Date"]]
num_years = max(years) - min(years) + 1
threshold = 30

model = ThresholdExcessModel(
    data=data["Rainfall"], threshold=threshold, num_years=num_years)

model.fit()
model.model_evaluation_plot()


#%%
round(len(model.extremes) / len(model.data), 5)

#%%
return_level = model.return_level(100)
{key: val.round(1) for (key, val) in return_level.items()}
#%%

model = PointProcesModel(data["Rainfall"], threshold, num_years=num_years)

model.fit()
model.model_evaluation_plot()
#%%



data = pd.read_csv("../data/rain.csv", parse_dates=[0])
years = [date.year for date in data["Date"]]
num_years = max(years) - min(years) + 1
threshold = 30
extr = data["Rainfall"][data["Rainfall"] > threshold] - threshold


extr = data["Rainfall"][data["Rainfall"] > threshold]


# # %%
# model = NHPP(num_years=num_years, threshold=threshold)
# model.fit(extr)
# # %%
# import numpy as np
# np.sqrt(model.covar)
# # %%
model = PointProcesModel(extr, threshold, num_years=num_years)

model.fit()
model.mle_model.theta

model.plot_diagnostics()
print(model.mle_model.theta)

#%%

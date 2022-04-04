#Feature selection algorithm class 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, f_regression
 

class FeatureSelector():
    def __init__(self, df, target_col):
        self.df = df.dropna()
        self.df = self.df.select_dtypes(['number'])
        print(np.where(self.df.var()==0))
        self.df = self.df.drop(columns = self.df.columns[np.where(self.df.var()==0)]) #Entire dataset
        self.target_col = target_col

        self.y = self.df[self.target_col] #Target column for prediction
        self.X_raw = self.df.loc[:, self.df.columns != self.target_col] #Unscaled X values
        self.X = Normalizer().fit_transform(self.X_raw) #Normalize the X values
        self.Xdf = pd.DataFrame(self.X)
        self.Xdf.columns = self.X_raw.columns
    

    #Mutual Information (classification)
    def mutual_info_class(self):
        mi = mutual_info_classif(self.X, self.y)
        #mi = StandardScaler().fit_transform(mi.reshape(-1,1))
        mi_df = pd.DataFrame(mi)
        mi_df['Feature'] = self.X_raw.columns
        mi_df.columns = ['Score','Feature']
        mi_df = mi_df.sort_values(by = 'Score', ascending = False).loc[:,('Feature','Score')].reset_index(drop=True)
        return mi_df

    #Mutual Information (regression)
    def mutual_info_regress(self):
        mi = mutual_info_regression(self.X, self.y)
        #mi = StandardScaler().fit_transform(mi.reshape(-1,1))
        mi_df = pd.DataFrame(mi)
        mi_df['Feature'] = self.X_raw.columns
        mi_df.columns = ['Score','Feature']
        mi_df = mi_df.sort_values(by = 'Score', ascending = False).loc[:,('Feature','Score')].reset_index(drop=True)
        return mi_df

    def mrmr(self):
        # compute F-statistics and correlations
        F = pd.Series(f_regression(self.Xdf, self.y)[0], index = self.Xdf.columns)
        corr = self.Xdf.corr().abs().clip(.00001) # minimum value of correlation set to .00001 (to avoid division by zero)

        # initialize list of selected features and list of excluded features
        selected = []
        not_selected = list(self.Xdf.columns)

        # repeat K times: 
        # compute FCQ score for all the features that are currently excluded,
        # then find the best one, add it to selected, and remove it from not_selected
        for i in range(len(self.Xdf.columns)):

            # compute FCQ score for all the (currently) excluded features (this is Formula 2)
            score = F.loc[not_selected] / corr.loc[not_selected, selected].mean(axis = 1).fillna(.00001)

            # find best feature, add it to selected and remove it from not_selected
            best = score.index[score.argmax()]
            selected.append(best)
            not_selected.remove(best)
        return selected

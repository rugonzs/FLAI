import pandas as pd
import bnlearn as bn
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
class Data():
    def __init__(self, data = None, transform = True, verbose = 0):
        """
        Initialize the Data class.

        Args:
        data (DataFrame, optional): The data to be used. If None, an exception is raised. Default is None.
        transform (bool, optional): If True, the data is transformed to numerical form. Default is True.
        verbose (int, optional): Verbosity level. Default is 0.
        """
        if data is None:
            raise Exception("Data is not provided")
        self.data = data
        self.transform = transform
        if self.transform:
            self.transform_data_numeric()

    def transform_data_numeric(self, verbose = 0):
        """
        Transform the data to numerical form.

        Args:
        verbose (int, optional): Verbosity level. Default is 0.
        """

        if self.data is None:
            raise Exception("Data is not provided")
        dfhot, dfnum = bn.df2onehot(self.data,verbose = verbose)
        #self.data2 = dfnum
        ### Add the transform map to comeback.
        enc = OrdinalEncoder()
        enc.fit(self.data)
        self.map_cat = enc.categories_
        self.data = pd.DataFrame(enc.transform(self.data), columns = self.data.columns)
    def fairness_metrics(self, target_column = None, predicted_column = None, 
                        columns_fair = None):
        """
        Calculate fairness for a subgroup of population.

        Args:
        target_column (str, optional): The target column. If None, an exception is raised. Default is None.
        predicted_column (str, optional): The predicted column. If None, an exception is raised. Default is None.
        columns_fair (list, optional): List of column names to consider for fairness. Default is None.
        """

        if target_column is None:
            raise Exception("target_column is not provided")
        if predicted_column is None:
            raise Exception("predicted_column is not provided")
        #Confusion Matrix
        result = {}
        result.update({'model' : self.metrics(target_column, predicted_column)})
        
        if not columns_fair is None:
            for c in columns_fair.keys():
                privileged = self.metrics(target_column, predicted_column, {'name' : c, 'value': columns_fair[c]['privileged']})
                unprivileged = self.metrics(target_column, predicted_column, {'name' : c, 'value': columns_fair[c]['unprivileged']})
                #disparate_impact = len(self.data[(self.data[predicted_column] == 1) & (self.data[c] == columns_fair[c]['unprivileged'])])  / len(self.data[(self.data[predicted_column] == 1) & (self.data[c] == columns_fair[c]['privileged'])]) 
                result.update({c: {'privileged' : privileged, 'unprivileged' : unprivileged,
                                   'fair_metrics' : {'EOD' : unprivileged['TPR'] - privileged['TPR'],
                                                     'DI' : unprivileged['PPP'] / privileged['PPP'],
                                                     'SPD' :  unprivileged['PPP'] - privileged['PPP'],
                                                     'OD' :  (unprivileged['FPR'] - privileged['FPR']) + (unprivileged['TPR'] - privileged['TPR']),
                                                     }}})

        return result
    def theil_index(self, y_true = None, y_pred = None, value_pred = None):
        """
        Calculate the Theil index for the prediction.

        Args:
        y_true (array, optional): The true labels. If None, an exception is raised. Default is None.
        y_pred (array, optional): The predicted labels. If None, an exception is raised. Default is None.
        value_pred (int, optional): The value to predict. If None, an exception is raised. Default is None.
        """

        if y_true is None:
            raise Exception("y_true is not provided")
        if y_pred is None:
            raise Exception("y_pred is not provided")
        if value_pred is None:
            raise Exception("value_pred is not provided")
        
        y_pred = y_pred.ravel()
        y_true = y_true.ravel()
        y_pred = (y_pred == value_pred).astype(np.float64)
        y_true = (y_true == value_pred).astype(np.float64)
        b = 1 + y_pred - y_true
        return np.mean(np.log((b / np.mean(b))**b) / np.mean(b))
           
    def metrics(self, target_column = None, predicted_column = None, column_filter = None):
        """
        Calculate various metrics for the prediction.

        Args:
        target_column (str, optional): The target column. If None, an exception is raised. Default is None.
        predicted_column (str, optional): The predicted column. If None, an exception is raised. Default is None.
        column_filter (dict, optional): Dictionary with keys as column names and values as filters. Default is None.
        """

        if target_column is None:
            raise Exception("target_column is not provided")
        if predicted_column is None:
            raise Exception("predicted_column is not provided")
        if column_filter is None:
            data = self.data
        else:
            print('Calculating metrics for :', column_filter['name'],' the value : ', column_filter['value'])
            data = self.data[self.data[column_filter['name']] == column_filter['value']]

        cm=confusion_matrix(data[target_column],data[predicted_column])
        ti = self.theil_index(data[target_column],data[predicted_column],1)
        TN, FP, FN, TP = cm.ravel()
        
        N = TP+FP+FN+TN #Total population
        ACC = (TP+TN)/N #Accuracy
        TPR = TP/(TP+FN) # True positive rate
        FPR = FP/(FP+TN) # False positive rate
        FNR = FN/(TP+FN) # False negative rate
        PPP = (TP + FP)/N # % predicted as positive
        return {'ACC' : ACC, 'TN' : TN, 'FP' : FP, 'FN' : FN, 'TP' : TP,'TPR' : TPR, 'FPR': FPR, 'FNR' : FNR, 'PPP' : PPP }
    def get_df_metrics(self, metrics_json = None):
        """
        Get the performance and fairness metrics as dataframes.

        Args:
        metrics_json (json, optional): The metrics in json format. If None, an exception is raised. Default is None.
        """
        if metrics_json is None:
            raise Exception("metrics_json is not provided")
        df_performance = pd.DataFrame(columns = ['ACC', 'TN', 'FP', 'FN', 'TP', 'TPR', 'FPR', 'FNR', 'PPP'])
        df_fairness = pd.DataFrame(columns = ['EOD', 'DI', 'SPD', 'OD'])
        for k in metrics_json.keys():
            if k == 'model':
                df_performance.loc['model'] = metrics_json['model'].values()
            else:
                df_performance.loc[k+'_privileged'] = metrics_json[k]['privileged'].values()
                df_performance.loc[k+'_unprivileged'] = metrics_json[k]['unprivileged'].values()
                df_fairness.loc[k+'_fair_metrics'] = metrics_json[k]['fair_metrics'].values()
        return df_performance,df_fairness
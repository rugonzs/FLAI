import pandas as pd
import bnlearn as bn
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import matplotlib.pyplot as plt
import math
plt.style.use("seaborn-dark-palette")

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
    
    def fairness_eqa_eqi(self, features = None, target_column = None, column_filter = None,plot = True):
        """
        Calculate fairness metrics for the data.

        Args:
        target_column (str, optional): The target column. If None, an exception is raised. Default is None.
        features (dict, optional): Dictionary with keys as column names as feature. Default is None.
        column_filter (dict, optional): Dictionary with keys as column names as sensible. Default is None.
        """

        if target_column is None:
            raise Exception("target_column is not provided")
        if features is None:
            raise Exception("features is not provided")
        if column_filter is None:
            raise Exception("predicted_column is not column_filter")
  
  
        df_aux = self.data.groupby(by=column_filter + features).agg({target_column: ['count', 'sum']})
        df_aux_ideal = self.data.groupby(by=features).agg({target_column: ['count', 'sum']})
        df_aux.columns = df_aux.columns.droplevel(0)
        df_aux = df_aux.reset_index()
        combinations_s = df_aux[column_filter].value_counts().index.values
        df_aux = df_aux.set_index(column_filter + features)

        df_aux_ideal.columns = df_aux_ideal.columns.droplevel(0)
        df_aux_ideal = df_aux_ideal.reset_index()
        combinations_f = df_aux_ideal[features].value_counts().index.values
        df_aux_ideal['px'] = df_aux_ideal['sum'] / df_aux_ideal['count']
        df_aux_ideal = df_aux_ideal.sort_values(by=['px']+features)
        df_aux_ideal = df_aux_ideal.set_index(features)
        df_aux_ideal['dx'] = [0] + (df_aux_ideal['count'].cumsum() / df_aux_ideal['count'].sum()).tolist()[:-1]
        df_aux['px'] = df_aux['sum'] / df_aux['count']


        n_group = combinations_s.shape[0]
        groups = [str(column_filter) + str(s) for s in combinations_s]
        combinations = [[s + f for s in combinations_s] for f in combinations_f]
        for c,f in zip(combinations,combinations_f):
            for n in range(n_group):
                if c[n] in df_aux.index:
                    df_aux_ideal.loc[f,'px_'+str(n)] = df_aux.loc[c[n]]['px']
                    df_aux_ideal.loc[f,'count_'+str(n)] = df_aux.loc[c[n]]['count']

                else:
                    df_aux_ideal.loc[f,'px_'+str(n)] = 0
                    df_aux_ideal.loc[f,'count_'+str(n)] = 0
                df_aux_ideal['dx_'+str(n)] = [0] + (df_aux_ideal['count_'+str(n)].cumsum() / df_aux_ideal['count_'+str(n)].sum()).tolist()[:-1]
                
        if plot:
            self.plot_fairness_eqa_eqi(df_aux_ideal,n_group,groups)
        n_p = -1
        p_max = 0
        d_max = 0
        for n in range(n_group):
            p_aux = df_aux_ideal['px_'+str(n)].max()
            d_aux = df_aux_ideal['dx_'+str(n)].max()
            if p_aux > p_max:
                p_max = p_aux
                d_max = d_aux
                n_p = n
            elif p_aux == p_max:
                if d_aux < d_max:
                    p_max = p_aux
                    d_max = d_aux
                    n_p = n

        df_f = pd.DataFrame(columns = ['group','reference','EQI','EQA','F'])
        for n in range(n_group):
            if n != n_p:
                eqi = (df_aux_ideal['dx_'+str(n_p)] - df_aux_ideal['dx_'+str(n)]).values
                eqa = (df_aux_ideal['px_'+str(n_p)] - df_aux_ideal['px_'+str(n)]).values

                EQI = np.round(eqi.mean(),2)
                EQA = np.round(eqa.mean(),2)
                F = np.round(math.sqrt(EQA**2 + EQI**2),2)
                df_f.loc[n] = [groups[n],groups[n_p],EQI,EQA,F]
        return df_f,df_aux_ideal
                
    def plot_fairness_eqa_eqi(self, df_aux_ideal,n_group,groups):
        fig = plt.figure(figsize=(20, 12))
        fig.subplots_adjust(hspace=0.8)
        gs = fig.add_gridspec(6,4)

        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax2 = fig.add_subplot(gs[0:2, 2:4])
        ax3 = fig.add_subplot(gs[2:6, :])

        ### plot dx
        ax1.set_title('Distribution',fontsize="20")
        ax1.set_xlabel('Feature Vector',fontsize="20")
        ax1.set_ylabel('EQI',fontsize="20")
        for n in range(n_group):
            ax1.plot(range(df_aux_ideal.shape[0]), 
                 df_aux_ideal['dx_'+str(n)], '-',linestyle='dashed',  marker='o',markersize=5,linewidth=2,label=groups[n])
        ax1.legend(fontsize="20")

        ### plot px
        ax2.set_title('Probability',fontsize="20")
        ax2.set_xlabel('Feature Vector',fontsize="20")
        ax2.set_ylabel('EQA',fontsize="20")
        for n in range(n_group):
            ax2.plot(range(df_aux_ideal.shape[0]), 
                 df_aux_ideal['px_'+str(n)], '-',linestyle='dashed',  marker='o',markersize=5,linewidth=2,label=groups[n])
        ax2.legend(fontsize="20")

        ### plot curve
        ax3.set_title('Fairness Curve',fontsize="20")
        ax3.set_xlabel('EQI',fontsize="20")
        ax3.set_ylabel('EQA',fontsize="20")
        for n in range(n_group):
            ax3.plot(df_aux_ideal['dx_'+str(n)], 
                 df_aux_ideal['px_'+str(n)], '-',linestyle='dashed',  marker='o',markersize=5,linewidth=2,label=groups[n])
        ax3.legend(fontsize="20")


        plt.show()
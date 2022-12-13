import pandas as pd
import bnlearn as bn
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
class Data():
    def __init__(self, data = None, transform = True, verbose = 0):
        if data is None:
            raise Exception("Data is not provided")
        self.data = data
        self.transform = transform
        if self.transform:
            self.transform_data_numeric()

    def transform_data_numeric(self, verbose = 0):
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
        """Calculate fairness for subgroup of population"""
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
                disparate_impact = len(self.data[(self.data[predicted_column] == 1) & (self.data[c] == columns_fair[c]['unprivileged'])])  / len(self.data[(self.data[predicted_column] == 1) & (self.data[c] == columns_fair[c]['privileged'])]) 
                result.update({c: {'privileged' : privileged, 'unprivileged' : unprivileged,
                                   'fair_metrics' : {'Equal_Opportunity_Difference' : unprivileged['TPR'] - privileged['TPR'],
                                                     'Disparate Impact' : unprivileged['PPP'] / privileged['PPP'],
                                                     'Statistical parity Difference' :  unprivileged['PPP'] - privileged['PPP']}}})

        return result
    
    def metrics(self, target_column = None, predicted_column = None, column_filter = None):
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
        TN, FP, FN, TP = cm.ravel()
        
        N = TP+FP+FN+TN #Total population
        ACC = (TP+TN)/N #Accuracy
        TPR = TP/(TP+FN) # True positive rate
        FPR = FP/(FP+TN) # False positive rate
        FNR = FN/(TP+FN) # False negative rate
        PPP = (TP + FP)/N # % predicted as positive
        return {'ACC' : ACC, 'TPR' : TPR, 'FPR': FPR, 'FNR' : FNR, 'PPP' : PPP}

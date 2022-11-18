import pandas as pd
import bnlearn as bn

class Data():
    def __init__(self, data = None, transform = True, verbose = 0):
        if data is None:
            raise Exception("Data is not provided")
        self.data = data
        self.transform = transform
        if self.transform:
            self.transform_data_numeric(self.data)

    def transform_data_numeric(self, data = None, verbose = 0):
        if data is None:
            raise Exception("Data is not provided")
        dfhot, dfnum = bn.df2onehot(data,verbose = verbose)
        
        ### Add the transform map to comeback.
        self.data = dfnum
        

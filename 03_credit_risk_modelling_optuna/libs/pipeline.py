import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from libs.evaluation import Metrics 
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


class Pipe():
    
    def __init__(self, df, model, target_var:str, test_size:float = 0.3, **kwargs):
        
        self.df = df
        self.target_var = target_var
        self.model = model 
        self.test_size = test_size 
        
        self.modelname = kwargs.get('modelname', None)
        self.cutmonth = kwargs.get('cutmonth', None)

        # Run everything
        # Grace Hopper ja pensava em subrotinas, que lindo
        self.runpipe()

    def runpipe(self):
        self.split()
        # Trainings the model
        self.model.fit(self.X_train, self.y_train)

        # Generate probabilities
        self.probs()

        self.metrics()

        self.overfitting = self.popin.scores - self.oos.scores
    def probs(self):

        self.y_pred_train = self.model.predict_proba(self.X_train)[:,1]
        self.y_pred_test = self.model.predict_proba(self.X_test)[:,1]

    def metrics(self):

        # pipemodel is a function which is responsible to run the pipe
        self.popin = Metrics(y_real = self.y_train,  
                            model_probs = self.y_pred_train)

        self.oos = Metrics(y_real = self.y_test,  
                            model_probs = self.y_pred_test)

        #return self.popin.scores, self.oos.scores
        #print('popin', self.popin.scores, 
        #      '\n oos', self.oos.scores, 
        #      '\n overfitting', self.overfitting)

    def split(self):
        
        self.X = self.df.drop(self.target_var, axis = 1)
        self.df.shape
        self.X.shape
        self.y = self.df[self.target_var]
        self.y.shape

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, 
                                                                                self.y,
                                                                                test_size = self.test_size)
    
    def convert_category(self):
        '''
        Converts object to categoric type for lightgbm
        '''
        dftypes = pd.DataFrame(self.df.dtypes).reset_index().rename(columns = {'index':'ft',0:'dtype'})
        objvars = list(dftypes[dftypes.dtype == 'object']['ft'])
        self.df[objvars] = self.df[objvars].astype('category')
                         
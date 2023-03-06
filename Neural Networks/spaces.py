import numpy as np
from hyperopt import hp

def CNN_space():
    space = {
    'numb_filt1': hp.choice('numb_filt1',[32, 64, 128]),
    'kernel_size1': hp.choice('kernel_size1',[3, 5, 7]),
    'pooling_size1': hp.choice('pooling_size1',[3, 5, 7]),
    'dropout1': hp.uniform('dropout1',0.0,0.6),

    'numb_filt2': hp.choice('numb_filt2',[32, 64, 128]),
    'kernel_size2': hp.choice('kernel_size2',[3, 5, 7]),
    'pooling_size2': hp.choice('pooling_size2',[3, 5, 7]),
    'dropout2': hp.uniform('dropout2',0.0,0.6),

    'lr': hp.loguniform('lr', -4*np.log(10), -3*np.log(10))
    }
    return space

def CNN_1_space():
    space = {
    'numb_filt1': hp.choice('numb_filt1',[32, 64, 128]),
    'kernel_size1': hp.choice('kernel_size1',[3, 5, 7]),
    'pooling_size1': hp.choice('pooling_size1',[3, 5, 7]),
    'dropout1': hp.uniform('dropout1',0.0,0.6),

    'lr': hp.loguniform('lr', -4*np.log(10), -3*np.log(10))
    }
    return space

def CNN_3_space():
    space = {
    'numb_filt1': hp.choice('numb_filt1',[32, 64, 128]),
    'kernel_size1': hp.choice('kernel_size1',[3, 5, 7]),
    'pooling_size1': hp.choice('pooling_size1',[3, 5, 7]),
    'dropout1': hp.uniform('dropout1',0.0,0.6),

    'numb_filt2': hp.choice('numb_filt2',[32, 64, 128]),
    'kernel_size2': hp.choice('kernel_size2',[3, 5, 7]),
    'pooling_size2': hp.choice('pooling_size2',[3, 5, 7]),
    'dropout2': hp.uniform('dropout2',0.0,0.6),
    
    'numb_filt3': hp.choice('numb_filt3',[32, 64, 128]),
    'kernel_size3': hp.choice('kernel_size3',[3, 5, 7]),
    'pooling_size3': hp.choice('pooling_size3',[3, 5, 7]),
    'dropout3': hp.uniform('dropout3',0.0,0.6),
    
    'lr': hp.loguniform('lr', -4*np.log(10), -3*np.log(10))
    }
    return space

def BiGRU_space():
    space = {
    'gruunits' : hp.choice('gruunits',[32, 64, 128]),
    'grurecdrop1' : hp.uniform('grurecdrop1',0.0, 0.6),
    
    'lr': hp.loguniform('lr',-4*np.log(10), -3*np.log(10))
    }
    return space

def ANConv1D_space():
    space = {
    'gruunits' : hp.choice('gruunits',[32, 64, 128]),
    'grurecdrop1' : hp.uniform('grurecdrop1',0.0, 0.6),

    'numb_filt1': hp.choice('numb_filt1',[32, 64, 128]),
    'kernel_size1': hp.choice('kernel_size1',[3, 5, 7]),
    'pooling_size1': hp.choice('pooling_size1',[3, 5, 7]),
    'dropout1': hp.uniform('dropout1',0.0,0.6),
    
    'lr': hp.loguniform('lr', -4*np.log(10), -3*np.log(10))
    }
    return space

def ANConv2D_space():
    space = {
    'gruunits' : hp.choice('gruunits',[32, 64, 128]),
    'grurecdrop1' : hp.uniform('grurecdrop1',0.0, 0.6),

    'numb_filt1': hp.choice('numb_filt1',[32, 64, 128]),
    'kernel_size1': hp.choice('kernel_size1',[3, 5, 7]),
    'pooling_size1': hp.choice('pooling_size1',[3, 5, 7]),
    'dropout1': hp.uniform('dropout1',0.0,0.6),
    
    'lr': hp.loguniform('lr', -4*np.log(10), -3*np.log(10))
    }
    return space

def AN2Conv1D_space():
    space = {
    'gruunits1' : hp.choice('gruunits1',[32, 64, 128]),
    'grurecdrop1' : hp.uniform('grurecdrop1',0.0, 0.6),

    'gruunits2' : hp.choice('gruunits2',[32, 64, 128]),
    'grurecdrop2' : hp.uniform('grurecdrop2',0.0, 0.6),

    'numb_filt1': hp.choice('numb_filt1',[32, 64, 128]),
    'kernel_size1': hp.choice('kernel_size1',[3, 5, 7]),
    'pooling_size1': hp.choice('pooling_size1',[3, 5, 7]),
    'dropout1': hp.uniform('dropout1',0.0,0.6),
    
    'lr': hp.loguniform('lr', -4*np.log(10), -3*np.log(10))
    }
    return space

def AN_space():
    space = {
    'gruunits' : hp.choice('gruunits',[32, 64, 128]),
    'grurecdrop1' : hp.uniform('grurecdrop1',0.0, 0.6),
    
    'lr': hp.loguniform('lr', -4*np.log(10), -3*np.log(10))
    }
    return space

def AN2_space():
    space = {
    'gruunits1' : hp.choice('gruunits1',[32, 64, 128]),
    'grurecdrop1' : hp.uniform('grurecdrop1',0.0, 0.6),

    'gruunits2' : hp.choice('gruunits2',[32, 64, 128]),
    'grurecdrop2' : hp.uniform('grurecdrop2',0.0, 0.6),
    
    'lr': hp.loguniform('lr', -4*np.log(10), -3*np.log(10))
    }
    return space

def AN3_space():
    space = {
    'gruunits1' : hp.choice('gruunits1',[32, 64, 128]),
    'grurecdrop1' : hp.uniform('grurecdrop1',0.0, 0.6),

    'gruunits2' : hp.choice('gruunits2',[32, 64, 128]),
    'grurecdrop2' : hp.uniform('grurecdrop2',0.0, 0.6),

    'gruunits3' : hp.choice('gruunits3',[32, 64, 128]),
    'grurecdrop3' : hp.uniform('grurecdrop3',0.0, 0.6),
    
    'lr': hp.loguniform('lr', -4*np.log(10), -3*np.log(10))
    }
    return space

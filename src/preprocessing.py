import pandas as pd 
from sklearn.preprocessing import MinMaxScaler,StandardScaler


def get_data(args):
    data = pd.read_csv(args.data)
    data = data.rename(columns={'Unnamed: 0': 'customer_ID'})
    X = data.drop(['customer_ID','_10_target_is_churn'],axis=1)
    y = data['_10_target_is_churn']
    return X,y 
        

def scale_method(args):
    if args.scaler == 'min_max':
        scaler = MinMaxScaler()
    if args.scaler == 'standard':
        scaler = StandardScaler()
    else:
        scaler = 'passthrough'
    
    return scaler 




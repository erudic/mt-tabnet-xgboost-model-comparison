from ast import literal_eval
import pandas as pd



def load(size,type='train_val'):
    df = pd.read_csv(f'../../data/{size}/{type}.csv')
    df = _parse_to_array(df,'Weather_Condition_Arr')
    df_X,df_Y = _split_features(df,'Severity')
    return df_X, df_Y

def _parse_to_array(df,column):
    df[column]=df['Weather_Condition_Arr'].apply(lambda x: literal_eval(x) if str(x)!='nan' else x)
    return df

def _split_features(df,target):
    return df[df.columns.drop(target)], df[target]
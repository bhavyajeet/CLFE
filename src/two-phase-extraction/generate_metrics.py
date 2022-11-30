import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
def get_metrics(average='weighted'):
    metric_dict={}
    bad_lang = ['as','pa','or','ml']
    for k in set(results['langs']):
        if k in bad_lang:
            continue
        y_true = [i for i,j in zip(results['originals'],results['langs']) if j==k]
        y_pred = [i for i,j in zip(results['preds'],results['langs']) if j==k]
        metric_dict[k] =[ accuracy_score(y_true, y_pred),
                         precision_score(y_true, y_pred,average=average),
                         recall_score(y_true, y_pred,average=average),
                         f1_score(y_true, y_pred,average=average)]
    df = pd.DataFrame(metric_dict)
    df['All'] = df.sum(axis=1)/df.shape[1]
    df = df.T.reset_index()
    df.columns=['lang','accuracy','precision','recall','f1']  
    return df


if __name__=="__main__":
    use_gold=False
    if use_gold==True:
        results = json.load(open('saved_metrics/9.json','r'))
        print(get_metrics())
    else:
        results = json.load(open('saved_metrics/pred0.json','r'))
        et = get_metrics()['f1']
        df = get_metrics()
        df['precision']*=0.54 # Precision from preprocessing script
        df['recall']*=0.77 # Recall from preprocessing script
        df['f1'] = df.apply(lambda x:(2*x[2]*x[3])/(x[2]+x[3]),axis=1)
        print(df)
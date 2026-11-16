"""Train an equipment failure classifier on reproducible synthetic sensor data."""
import joblib, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
def make_data(n=4000,seed=7):
    rng=np.random.default_rng(seed)
    temp=rng.normal(70,12,n).clip(25,120)
    vibration=rng.gamma(2,2,n).clip(0,25)
    pressure=rng.normal(5.5,1,n).clip(1,10)
    hours=rng.integers(0,30000,n)
    risk=(temp>95)*1.1+(vibration>9)*1.5+(pressure<3.5)*1.0+(hours>22000)*.8
    y=rng.binomial(1,1/(1+np.exp(-(risk-2.0))))
    return np.column_stack([temp,vibration,pressure,hours]),y
def train():
    X,y=make_data()
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=7,stratify=y)
    model=RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=7).fit(Xtr,ytr)
    pred=model.predict(Xte)
    print(classification_report(yte,pred,zero_division=0))
    print("ROC-AUC:",round(roc_auc_score(yte,model.predict_proba(Xte)[:,1]),4))
    joblib.dump(model,"maintenance_model.joblib")
if __name__=="__main__": train()

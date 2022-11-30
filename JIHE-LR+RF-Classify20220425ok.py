#!/usr/bin/python
# coding:utf-8
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from mlxtend.classifier import StackingClassifier,StackingCVClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict
from collections import Counter
from sklearn import metrics
from sklearn.metrics import auc,roc_auc_score,roc_curve,precision_recall_curve,confusion_matrix
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.feature_selection import chi2,f_classif,SelectKBest 
from sklearn.model_selection import GridSearchCV
import shap
shap.initjs()
from matplotlib import pyplot as plt

def read_fea_csv_fast(input_path):
    input_path=open(input_path,'r')
    data=pd.read_csv(input_path,index_col=None,header=None)
    Y = data.values[:,0]; X = data.values[:,1:]
    print('Feature shape:',X.shape,'Sample dsitribution',Counter(Y.reshape(-1)),file=f1)
    return (X,Y)


def training_process(classifier,X,Y):
    Y_pred_proba = cross_val_predict(clf, X, Y, method="predict_proba", cv=5)  # Cross validation  
    print('Cross-validation results:', file=f1)
    Metrics1(Y,Y_pred_proba)
    return

def testing_process(clf, X1, Y1, X2, Y2):
    clf.fit(X1,Y1)
    Y2_pred_proba = clf.predict_proba(X2)
    print('Indepdent results:', file=f1)
    Metrics1(Y2, Y2_pred_proba)
    return

def Metrics1(Y,Y_pred_proba):
    Y_pred = np.zeros(Y.shape[0], );
    for i in range(Y.shape[0]):
        Y_pred[i] = 0 if Y_pred_proba[i, 1] < 0.5 else 1
    mat = metrics.confusion_matrix(Y, Y_pred, labels=[1, 0])   # the mat is ordered by increasing labels, thus giving the common orders
    FPR, TPR, threshold = roc_curve(Y, Y_pred_proba[:, 1], pos_label=1)  # Ytest_pred_proba[:,1] means the second column(i.e. the proba of label 1)
    PRE, RE, thres_pr = precision_recall_curve(Y, Y_pred_proba[:, 1])
    Metrics2(mat, FPR, TPR, RE, PRE)
    return

def Metrics2(mat,ROC1,ROC2,PR1,PR2):  #with PR curve
    TP = mat[0,0];FN = mat[0,1];FP = mat[1,0]; TN = mat[1,1]
    Sn = TP/(TP+FN);Sp=TN/(TN+FP)
    Acc = (TP+TN)/(TP+FN+FP+TN)
    MCC = ((TP*TN)-(FP*FN))/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)+0.00001)**0.5
    Pre = TP/(TP+FP+0.00001)
    F1=2*(Sn*Pre)/(Pre+Sn+0.00001)
    au_ROC = auc(ROC1,ROC2)
    au_PR = auc(PR1,PR2)
    print('FN: %i; TP+FN: %i; FP: %i; FP+TN: %i; Sn: %.4f; Sp: %.4f; Acc: %.4f; MCC: %.4f; Pre: %.4f;  F1: %.4f; au_ROC: %.4f; au_PR: %.4f'%(FN, TP+FN, FP, FP+TN, Sn, Sp, Acc, MCC, Pre, F1, au_ROC,au_PR),file=f1)
    print('ROC1','ROC',file=f2)
    for i in range(len(ROC1)):
        print(round(ROC1[i],3),round(ROC2[i],3),file=f2)
    print('PR1','PR2',file=f2)
    for i in range(len(PR1)):
        print(round(PR1[i],3),round(PR2[i],3),file=f2)
    return

def stack_model(X1,Y1,X2,Y2):
    clf1 = LogisticRegression()
    clf2 = RandomForestClassifier(n_estimators=100,random_state=100)
    lr = LogisticRegression()
    sclf1 = StackingCVClassifier(classifiers=[clf1, clf2],meta_classifier=lr)
    sclf2 = StackingClassifier(classifiers=[clf1, clf2],use_probas=True,meta_classifier=lr)
    skf = list(StratifiedKFold(n_splits=cv, random_state=0, shuffle=True).split(X1, Y1))
    print('Stacking model:',file=f1);print('Stacking model:',file=f2)
    for clf,label in zip([clf1,clf2,sclf1,sclf2],['LR','RF','StackingClassifier','StackingClassifier using proba']):
        print(label,file=f1);print(label,file=f2)
        for i,(train,val) in enumerate(skf):
            X1_train,Y1_train=X1[train],Y1[train];X1_val,Y1_val=X1[val],Y1[val]
            clf.fit(X1_train,Y1_train)
            Y1_val_proba=clf.predict_proba(X1_val)
            ''' recording the results of each fold
            print('Fold', i + 1, 'Results:', file=f1);print('Fold', i, 'Results:', file=f2);
            Metrics1(Y1_val,Y1_val_proba);
            '''
            Y1_all_proba = Y1_val_proba if i==0 else np.concatenate((Y1_all_proba,Y1_val_proba),axis=0)
            Y1_all=Y1_val.reshape(-1,1) if i==0 else np.concatenate((Y1_all,Y1_val.reshape(-1,1)),axis=0)            
        print(cv,'-CV results:',file=f1);print(cv,'-CV results:',file=f2);
        Metrics1(Y1_all, Y1_all_proba);
        print('Inde Results:',file=f1);print('Inde Results:',file=f2)
        clf.fit(X1,Y1)
        Y2_pred_proba = clf.predict_proba(X2)
        Metrics1(Y2, Y2_pred_proba)
    return


def training_process(classifier,X,Y):
    Y_pred_proba = cross_val_predict(clf, X, Y, method="predict_proba", cv=5)
    print('Cross-validation results:', file=f1)
    Metrics1(Y,Y_pred_proba)
    return


def fea_selection(X1,Y1,X2,Y2):
    selection = SelectKBest(select_methods, k=X1.shape[1])
    X1_select = selection.fit_transform(X1, Y1)   # data[columns]是特征矩阵，data['label']是标签数据
    X2_select = selection.transform(X2)
    record=open('Feature_selection.txt','w')
    print('feature selection:methods', select_methods,file=record )
    pd.DataFrame(selection.scores_).to_csv(record,header=['scores'],index=None,float_format='%.2f')
    pd.DataFrame(selection.pvalues_).to_csv(record,header=['pcalues'],index=None,float_format='%.2f')
    return X1_select,X2_select

if __name__=="__main__":
    train_feas='training_set.csv'
    test_feas='testing_set.csv'
    cv=5   #k fold cross validation
    f1=open('Result.txt','w');f2=open('ROC-PR-curves.txt','w')
    Xtrain,Ytrain=read_fea_csv_fast(train_feas)
    Xtest,Ytest=read_fea_csv_fast(test_feas)
    clf = RandomForestClassifier(n_estimators=100, random_state=100)  #define the classifier named as clf
    print('Cross validation:', file=f2)
    
    # Common ML methods
    ''' 
    print('ML process:',file=f1)
    training_process(clf,Xtrain,Ytrain)
    print('Indepedent:', file=f2)
    testing_process(clf, Xtrain, Ytrain, Xtest,Ytest)
    stack_model(Xtrain, Ytrain, Xtest, Ytest)   #stack model
    '''
    
    
    #Feature selection process
    #''' 
    f3 = open('Featuer_selection.results','w')
    select_methods=f_classif;
    top_feas_min,top_feas_max,step=115,135,1  #minimum and maximum number of selected features as well as feature step
    Xtrain,Xtest=fea_selection(Xtrain,Ytrain,Xtest,Ytest)
    top_feas_min,top_feas_max,step=[numfea_all,numfea_all+1,100] if top_feas_min==None else [top_feas_min,top_feas_max,step]
    for top_feas in np.arange(top_feas_min,top_feas_max,step):   #to perform feature selection
        print('Top',top_feas, 'features are consisered',file=f1);
        Xtrain_sel=Xtrain[:,:top_feas];Xtest_sel=Xtest[:,:top_feas]
        training_process(clf,Xtrain_sel,Ytrain)
        print('Indepedent:', file=f2)
        testing_process(clf, Xtrain_sel, Ytrain, Xtest_sel,Ytest)
        stack_model(Xtrain_sel, Ytrain, Xtest_sel, Ytest)
    #'''
    
    f1.close();f2.close()

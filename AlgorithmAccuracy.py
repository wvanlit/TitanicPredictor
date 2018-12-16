'''
#Project: Titanic Survival Predictor
#Author: Wessel van Lit
#Description: A program that tries to predict if a person surives on the Titanic based on certain factors like Age, Sex, Class, etc.
The program uses Machine Learning to calculate its predictions.
'''
# Import the libraries needed for this program
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import VotingClassifier


from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import warnings


# Terminal Setup
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.expand_frame_repr', False)
# Add File Paths for the datasets

TestFilePath = '/home/wessel/Documents/PetProjects/Data Science & Machine Learning/PythonPrograms/Kaggle Competitions/Titanic Competition/TitanicPredictor/Data/test.csv'
TrainFilePath = '/home/wessel/Documents/PetProjects/Data Science & Machine Learning/PythonPrograms/Kaggle Competitions/Titanic Competition/TitanicPredictor/Data/train.csv'

# Assign the datasets to variables
testData = pd.read_csv(TestFilePath)
trainData = pd.read_csv(TrainFilePath)

# Create X & y for predicting the surival of passengers
cols_to_use = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']
X = trainData[cols_to_use]
y = trainData.Survived
predictX = testData[cols_to_use]

# Create Usable Data with OneHotEncoding
one_hot_encoded_predict_X = pd.get_dummies(predictX)
one_hot_encoded_X = pd.get_dummies(X)
final_X, final_predict_X = one_hot_encoded_X.align(
    one_hot_encoded_predict_X, join='left', axis=1)

# Create pipelines for different models
RFtotalscore, GBtotalscore, LRtotalscore, KNtotalscore, DTtotalscore, XGBtotalscore, SVtotalscore, GPtotalscore, VCtotalscore = 0, 0, 0, 0, 0, 0, 0, 0, 0
for rs in range(1, 3):
    RFclassifier = make_pipeline(
        SimpleImputer(), RandomForestClassifier(random_state=rs))
    GBclassifier = make_pipeline(
        SimpleImputer(), GradientBoostingClassifier(random_state=rs))
    LRclassifier = make_pipeline(
        SimpleImputer(), LogisticRegression(random_state=rs))
    KNclassifier = make_pipeline(SimpleImputer(), KNeighborsClassifier())
    DTclassifier = make_pipeline(
        SimpleImputer(), DecisionTreeClassifier(random_state=rs))
    XGBclassifier = make_pipeline(
        SimpleImputer(), XGBClassifier(random_state=rs))
    SVclassifier = make_pipeline(SimpleImputer(), SVC(random_state=rs))
    GPclassifier = make_pipeline(
        SimpleImputer(), GaussianProcessClassifier(random_state=rs))

    VC = make_pipeline(SimpleImputer(), VotingClassifier(
        estimators=[('rf', RFclassifier), ('gb', GBclassifier), ('XGB', XGBclassifier)]))
    # Calculate Cross Validation for pipelines
    scores = cross_val_score(RFclassifier, final_X, y,
                             scoring='balanced_accuracy')
    RFscore = scores.mean()
    RFtotalscore += RFscore

    scores = cross_val_score(GBclassifier, final_X, y,
                             scoring='balanced_accuracy')
    GBscore = scores.mean()
    GBtotalscore += GBscore

    scores = cross_val_score(LRclassifier, final_X, y,
                             scoring='balanced_accuracy')
    LRscore = scores.mean()
    LRtotalscore += LRscore

    scores = cross_val_score(KNclassifier, final_X, y,
                             scoring='balanced_accuracy')
    KNscore = scores.mean()
    KNtotalscore += KNscore

    scores = cross_val_score(DTclassifier, final_X, y,
                             scoring='balanced_accuracy')
    DTscore = scores.mean()
    DTtotalscore += DTscore

    scores = cross_val_score(XGBclassifier, final_X, y,
                             scoring='balanced_accuracy')
    XGBscore = scores.mean()
    XGBtotalscore += XGBscore

    scores = cross_val_score(SVclassifier, final_X, y,
                             scoring='balanced_accuracy')
    SVscore = scores.mean()
    SVtotalscore += SVscore

    scores = cross_val_score(GPclassifier, final_X, y,
                             scoring='balanced_accuracy')
    GPscore = scores.mean()
    GPtotalscore += GPscore

    scores = cross_val_score(VC, final_X, y, scoring='balanced_accuracy')
    VCscore = scores.mean()
    VCtotalscore += VCscore

# Print Accuracy
print('RF Accuracy:', round((RFtotalscore / rs * 100), 2), '%')
print('GB Accuracy:', round((GBtotalscore / rs * 100), 2), '%')
print('LR Accuracy:', round((LRtotalscore / rs * 100), 2), '%')
print('KN Accuracy:', round((KNtotalscore / rs * 100), 2), '%')
print('DT Accuracy:', round((DTtotalscore / rs * 100), 2), '%')
print('XGB Accuracy:', round((XGBtotalscore / rs * 100), 2), '%')
print('SV Accuracy:', round((SVtotalscore / rs * 100), 2), '%')
print('GP Accuracy:', round((GPtotalscore / rs * 100), 2), '%')
print('VC Accuracy:', round((VCtotalscore / rs * 100), 2), '%')

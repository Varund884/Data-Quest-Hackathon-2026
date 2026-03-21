import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import joblib
import shap
#Make a dataframe from the first dataset 
Dataframe = pd.read_csv('./Datasets/Dataset1.csv')
#Let's see how big the data is 
from sklearn.model_selection import train_test_split



print("Shape:", Dataframe.shape)
#The column names of the dataset
print("\nColumns:", Dataframe.columns.tolist())
#Print the first five player rows
print("\n first 5 rows:", Dataframe.head())



#Count all the true values on each column thats null
Dataframe.isnull.sum() #Everything is 0 


#Count how many injuries
print(Dataframe['Injury_Next_Season'].value_counts())
#See how many different positions there are
print(Dataframe['Position'].value_counts())

#Change the position from text to numbers so ML model knows
position_convertion = {
    'Midfielder': 2,
    'Defender': 1,
    'Forward': 3,
    'Goalkeeper': 0
}
Dataframe['Position'] = Dataframe['Position'].map(position_convertion)


#See if it works with the new Values 
print(Dataframe['Position'].value_counts())
#Check if any new null appeared 
print(Dataframe['Position'].isnull().sum())


# We want to split the result variable vs Non Result 
NonResult = Dataframe.drop('Injury_Next_Season', axis=1)
Result = Dataframe['Injury_Next_Season']

print("Result Shape", Result.shape)
print("NonResult Shape:", NonResult.shape)


#Split into Test(Majority of people) and Train 
#Split into Test(Minority) and Train 
NonResult_train, NonResult_test, Result_train, Result_test = train_test_split(
    NonResult, Result,
    test_size= 0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    min_sample_split=5,
    class_weight="balanced"
)

model.fit(NonResult_train,Result_train)


Result_pred = model.predict(NonResult_test)
Result_prob = model.predict_proba(NonResult_test)[:, 1]
 # Evaluating the model. This will we be for how many we get right from the test set. 

print(f"Accuracy:  {accuracy_score(Result_test, Result_pred):.3f}")
print(f"F1 Score:  {f1_score(Result_test, Result_pred):.3f}")
print(f"ROC-AUC:   {roc_auc_score(Result_test, Result_prob):.3f}") # most likely and lest likely to get injured

# This saves everything the model learned into a file so we dont retrain everytime we want to make a prediction. We can load this. file later and use it to predict injuries for new players. This is important because training the model can take a long time, and we want to be able to make predictions quickly.
# our back end will load this file later for predictions.
joblib.dump(model, "injury_model.pkl")

# creating the explainer this teaches shap how our random forest model works. 
explainer = shap.TreeExplainer(model)
# This determines how each column affects the prediction of the model. 
shap_values = explainer.shap_values(NonResult_test)
# This creates a summary plot that shows the 
shap.summary_plot(shap_values[1], NonResult_test, plot_type="bar")















train = pd.read_csv('train_features.csv')  
test = pd.read_csv('test_features.csv')

features = ['total_bids','total_auctions','total_devices','no_of_ips']
target = ['outcome']
X = np.array(train[features])
y = np.array(train[target]).ravel()
X_testing = np.array(test[features])

ncols = len(train.columns)
nrows = len(train.index)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20 )
print(X_train)
print(y_train)

# XGBoost Model
XGB_Model = xgboost.XGBClassifier(max_depth=6,learning_rate=0.01,n_estimators=240,reg_alpha=0,reg_lambda=0)

# XGBoost Kfold Validation
kfold = KFold(n_splits=5)
result = cross_validate(XGB_Model, X_train, y_train, cv=kfold,scoring= ['accuracy','roc_auc'])

# XGBoost Accuracy
print("Accuracy :", result['test_accuracy'].mean())
print("ROC Accuracy:", result['test_roc_auc'].mean())

shp = y_test.reshape(y_test.size,1)
y_changed = np.hstack((1-shp,shp))

#XGBoost Test
XGB_Model.fit(X_train,y_train)
pred = XGB_Model.predict_proba(X_test)
pred1 = XGB_Model.predict(X_test)
acc = accuracy_score(y_test,pred1)
roc = roc_auc_score(y_changed,pred)
print('Test Accuracy',acc)
print('Test ROC Accuracy',roc)

# RandomForest Model
RF_Model = RandomForestClassifier(max_depth=7, random_state=None)

# RandomForest Kfold Validation
kfold = KFold(n_splits=5)
results = cross_validate(RF_Model, X_train, y_train, cv=kfold,scoring= ['accuracy','roc_auc'])

# RandomForest Accuracy
print("Accuracy :", results['test_accuracy'].mean())
print("ROC Accuracy:", results['test_roc_auc'].mean())

shp = y_test.reshape(y_test.size,1)
y_changed = np.hstack((1-shp,shp))
# RandomForest Test
RF_Model.fit(X_train,y_train)
pred = RF_Model.predict_proba(X_test)
pred1 = RF_Model.predict(X_test)
acc = accuracy_score(y_test,pred1)
roc = roc_auc_score(y_changed,pred)
print('Test Accuracy',acc)
print('Test ROC Accuracy',roc)

# Probability Prediction on Testing Data
XGB_Model.fit(X, y)
testing_prediction = XGB_Model.predict_proba(X_testing)[:,1]
test['prediction_probability'] = testing_prediction
test['prediction'] = 0
test.loc[test['prediction_probability'] > 0.5, 'prediction'] = 1
test.loc[test['prediction_probability'] <= 0.5, 'prediction'] = 0
test[['bidder_id','prediction_probability','prediction']].to_csv('Prediction.csv', sep=',', header=True, index=False)
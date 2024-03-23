import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle
 

def get_clean_data():
    data = pd.read_csv('data.csv')
    # clean the data
    data=data.drop(['Unnamed: 32','id'],axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
    print(data.info())
    return data

def create_model(data):
    x=data.drop('diagnosis',axis=1)  # Independent variable
    y=data['diagnosis'] #dependent variable
    
    # scale the data   
    scaler = StandardScaler()
    x=scaler.fit_transform(x)
    
    # split the data
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)
    
    # train the data
    model = LogisticRegression()
    model.fit(x_train,y_train)
    
    # test the model
    y_predict = model.predict(x_test)
    print('Accuracy of model :',accuracy_score(y_test,y_predict))
    print('Classification report :',classification_report(y_test,y_predict))
    
    return model,scaler

def main():
    data = get_clean_data()
    model,scaler = create_model(data)
    
    with open('model.pkl','wb') as f:
        pickle.dump(model,f)
        
    with open('scaler.pkl','wb') as f:
        pickle.dump(scaler,f)
    
if __name__=='__main__':
    main()


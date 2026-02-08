import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle

def train():
    # Load the data
    df = pd.read_csv('advertising.csv')

    # Split features and target
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']

    # 1. Initialize and fit the scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 2. Train the model on scaled data
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save both files
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Success: scaler.pkl and model.pkl have been created.")

if __name__ == "__main__":
    train()
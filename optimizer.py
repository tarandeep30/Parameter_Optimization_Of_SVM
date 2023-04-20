import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rnd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
best_accuracy = 0
cv_accuracy_list = []
sample_list = []


data = pd.read_csv('./ai4i2020.csv')
df = data.iloc[:, 2:9]

# Encoding type values
encoder = {'L': 1, 'M': 2, 'H': 3}
for i in range(df.shape[0]):
    df['Type'][i] = encoder[df['Type'][i]]


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


def get_accuracy(X_train, y_train, X_test, y_test, params):
    """Calculates the accuracy of an SVM model with given hyperparameters."""
    kernel, C, gamma, degree = params
    svc = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)    
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    return accuracy_score(y_test, y_pred)


def run_sample(X, y, iterations=100):
    """Trains and tests an SVM model on a random sample of the data."""
    global best_accuracy
    sample_best_accuracy = 0
    sample_best_gamma = 0     # rbf, poly, and sigmoid only
    sample_best_kernel = ''
    sample_best_C =  0       
    sample_best_degree = 0    # 1-5 for poly only
    sample_accuracy_list = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for _ in range(iterations):
        kernel = rnd.choice(kernel_list)
        C = rnd.randint(1, 7)
        gamma = rnd.randint(-1, 7)
        degree = rnd.randint(1, 7)
        if gamma < 1:
            gamma = rnd.choice(['scale', 'auto'])
        if kernel == 'poly':
            gamma = rnd.choice(['scale', 'auto'])

        accuracy = get_accuracy(X_train, y_train, X_test, y_test, [kernel, C, gamma, degree])

        if accuracy > sample_best_accuracy:
            sample_best_accuracy = accuracy
            sample_best_C = C
            sample_best_degree = degree
            sample_best_kernel = kernel
            sample_best_gamma = gamma
        
        sample_accuracy_list.append(sample_best_accuracy)
    
    sample_list.append([sample_best_kernel, sample_best_C, sample_best_gamma, sample_best_degree, sample_best_accuracy])
    
    if sample_best_accuracy > best_accuracy:
        global cv_accuracy_list
        cv_accuracy_list = sample_accuracy_list
        best_accuracy = sample_best_accuracy


for _ in range(10):
    run_sample(X, y, iterations=500)

sample_df = pd.DataFrame(sample_list, columns=['Kernel', 'C', 'Gamma', 'Degree', 'Accuracy'])
print(sample_df)

sample_df.to_csv('./result.csv', index=False)
sample_df.to_markdown('./result.md', index=False)

plt.plot(np.arange(len(cv_accuracy_list)), cv_accuracy_list)
plt.title('Convergence graph of the best SVM')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.show()

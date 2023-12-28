import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_absolute_error

# Load data
df = pd.read_csv('https://gist.github.com/netj/8836201.js"')

# Split data into features and target
X = df.drop(['species', 'petal_length'], axis=1)
y_class = df['species']
y_regr = df['petal_length']

# Split data into training and testing sets
X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=0)
X_train, X_test, y_regr_train, y_regr_test = train_test_split(X, y_regr, test_size=0.2, random_state=0)

# Train the classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_class_train)

# Make predictions on the test data and calculate accuracy
y_class_pred = clf.predict(X_test)
accuracy = accuracy_score(y_class_test, y_class_pred)
print("Classifier Accuracy: {:.2f}%".format(accuracy*100))

# Train the regressor
regr = RandomForestRegressor(n_estimators=100)
regr.fit(X_train, y_regr_train)

# Make predictions on the test data and calculate mean absolute error
y_regr_pred = regr.predict(X_test)
mae = mean_absolute_error(y_regr_test, y_regr_pred)
print("Regressor Mean Absolute Error: {:.2f}".format(mae))

# Train the clusterer
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Get the cluster labels
labels = kmeans.labels_

# Plot the data points with different colors based on their cluster label
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
plt.show()

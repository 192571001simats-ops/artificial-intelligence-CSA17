from sklearn.tree import DecisionTreeClassifier

hours = [[1], [2], [3], [4], [5], [6]]
results = ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass"]

model = DecisionTreeClassifier(random_state=0)
model.fit(hours, results)

value = float(input("Enter study hours: "))
prediction = model.predict([[value]])

print("Predicted result:", prediction[0])

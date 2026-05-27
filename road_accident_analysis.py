import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# LOADing DATA
df = pd.read_excel("road_accident_dataset.xlsx")


# BASIC ANALYSIS

average_speed = np.mean(df["Speed"])
print("Average Speed using NumPy:", average_speed)

print("\nSeverity distribution:")
print(df["Severity"].value_counts())

print("\nFirst 5 rows:")
print(df.head())

print("\nData Info:")
print(df.info())

print("\nStatistics:")
print(df.describe())


# 1. SEVERITY GRAPH

severity_counts = df["Severity"].value_counts()

plt.figure()
plt.bar(severity_counts.index, severity_counts.values)
plt.title("Accident Severity Distribution")
plt.xlabel("Severity")
plt.ylabel("Number of Accidents")
plt.savefig("severity_graph.png")
plt.show()


# 2. LOCATION GRAPH

location_counts = df["Location"].value_counts()

print("\nAccident Locations:")
print(location_counts)

plt.figure()
location_counts.plot(kind="bar")
plt.title("Accident Hotspots by Location")
plt.xlabel("Location")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("location_graph.png")
plt.show()


# 3. WEATHER GRAPH

weather_counts = df.groupby("Weather")["Severity"].count()

print("\nAccidents by Weather:")
print(weather_counts)

plt.figure()
weather_counts.plot(kind="bar")
plt.title("Accidents by Weather Condition")
plt.xlabel("Weather")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("weather_graph.png")
plt.show()


# 4. SPEED VS SEVERITY

speed_by_severity = df.groupby("Severity")["Speed"].mean()

print("\nAverage Speed by Severity:")
print(speed_by_severity)

plt.figure()
speed_by_severity.plot(kind="bar")
plt.title("Average Speed by Accident Severity")
plt.xlabel("Severity")
plt.ylabel("Average Speed")
plt.tight_layout()
plt.savefig("speed_severity_graph.png")
plt.show()


# 5. VEHICLES VS CASUALTIES

vehicles_vs_casualties = df.groupby("Vehicles")["Casualties"].mean()

print("\nAverage Casualties by Vehicles:")
print(vehicles_vs_casualties)

plt.figure()
vehicles_vs_casualties.plot(kind="bar")
plt.title("Vehicles vs Casualties")
plt.xlabel("Number of Vehicles")
plt.ylabel("Average Casualties")
plt.tight_layout()
plt.savefig("vehicles_casualties_graph.png")
plt.show()


# MACHINE LEARNING MODEL

df_ml = df.copy()

le_weather = LabelEncoder()
le_road = LabelEncoder()
le_severity = LabelEncoder()

df_ml["Weather"] = le_weather.fit_transform(df_ml["Weather"])
df_ml["Road_Condition"] = le_road.fit_transform(df_ml["Road_Condition"])
df_ml["Severity"] = le_severity.fit_transform(df_ml["Severity"])

X = df_ml[["Speed", "Vehicles", "Weather", "Road_Condition"]]
y = df_ml["Severity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)


# FEATURE IMPORTANCE

importance = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

print("\nFeature Importance:")
print(feature_importance_df.sort_values(by="Importance", ascending=False))
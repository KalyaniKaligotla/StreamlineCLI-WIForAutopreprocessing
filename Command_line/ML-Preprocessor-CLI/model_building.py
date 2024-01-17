# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score


class ModelBuilder:
    classification_algorithms = [
        ('Logistic Regression', LogisticRegression()),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Random Forest', RandomForestClassifier()),
        ('K-Nearest Neighbors', KNeighborsClassifier())
    ]

    regression_algorithms = [
        ('Linear Regression', LinearRegression()),
        ('Decision Tree', DecisionTreeRegressor()),
        ('Random Forest', RandomForestRegressor()),
        ('K-Nearest Neighbors', KNeighborsRegressor())
    ]

    @staticmethod
    def build_classification_model(data, target_variable):
        print("\nClassification Algorithms:")
        for i, (algorithm_name, _) in enumerate(ModelBuilder.classification_algorithms, 1):
            print(f"{i}. {algorithm_name}")

        while True:
            try:
                choice = int(input("\nEnter the number of the algorithm to build the model (Press -1 to go back): "))
                if choice == -1:
                    break
                elif 1 <= choice <= len(ModelBuilder.classification_algorithms):
                    _, model = ModelBuilder.classification_algorithms[choice - 1]
                    ModelBuilder.build_model(data, model, "Classification", target_variable)
                    break  # Exit loop after successful input
                else:
                    print("Invalid choice. Try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    @staticmethod
    def build_regression_model(data, target_variable):
        print("\nRegression Algorithms:")
        for i, (algorithm_name, _) in enumerate(ModelBuilder.regression_algorithms, 1):
            print(f"{i}. {algorithm_name}")

        while True:
            try:
                choice = int(input("\nEnter the number of the algorithm to build the model (Press -1 to go back): "))
                if choice == -1:
                    break
                elif 1 <= choice <= len(ModelBuilder.regression_algorithms):
                    _, model = ModelBuilder.regression_algorithms[choice - 1]
                    ModelBuilder.build_model(data, model, "Regression", target_variable)
                    break  # Exit loop after successful input
                else:
                    print("Invalid choice. Try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    @staticmethod
    def build_model(data, model, task, target_variable):
        print(f"\nBuilding {task} Model\U0001F447\n")

        # Check if the target column is present in the DataFrame
        if target_variable not in data.columns:
            print(f"Error: '{target_variable}' not found in the dataset. Unable to build the model.")
            return

        # Assuming 'target_variable' is the target variable
        y = data[target_variable]

        # Drop the target variable from the features
        X = data.drop(columns=[target_variable])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = model.predict(X_test)

        if task == "Regression":
            # Evaluate the regression model
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            print(f"\nMean Squared Error: {mse}")
            print(f"\nR-squared Score: {r2}")
            print(f"\nAccuracy: {100*r2}")

        elif task == "Classification":
            # Evaluate the classification model
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')
            f1 = f1_score(y_test, predictions, average='weighted')
            classification_rep = classification_report(y_test, predictions)
            confusion_mat = confusion_matrix(y_test, predictions)

            print(f"\nAccuracy: {accuracy:.2f*100}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1 Score: {f1:.2f}")
            print("\nClassification Report:\n", classification_rep)
            print("\nConfusion Matrix:\n", confusion_mat)

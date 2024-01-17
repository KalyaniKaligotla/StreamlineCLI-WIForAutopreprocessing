from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify, session
import pandas as pd
from io import StringIO, BytesIO
from flask import get_flashed_messages, jsonify, json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, accuracy_score, confusion_matrix,f1_score,mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.inspection import permutation_importance
from pandas.api.types import is_numeric_dtype
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure secret key

# Initialize the dataset as an empty DataFrame
dataset = pd.DataFrame()
global_model = None
global_X_test = None
global_y_test = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global dataset
    file = request.files['file']
    if file and file.filename.endswith('.csv'):
        dataset = pd.read_csv(file)
        return redirect(url_for('columns'))
    else:
        return "Invalid file format. Please upload a CSV file."

@app.route('/columns')
def columns():
    if not dataset.empty:
        columns = dataset.columns.tolist()
        return render_template('columns.html', columns=columns)
    else:
        return redirect(url_for('index'))

@app.route('/remove_column', methods=['POST'])
def remove_column():
    global dataset
    selected_column = request.form['column_to_remove']

    try:
        if selected_column in dataset.columns:
            dataset = dataset.drop(columns=[selected_column])
            response = {"success": True, "message": f"Column '{selected_column}' removed successfully."}
        else:
            response = {"success": False, "message": "Error: Column not found in the dataset."}

    except Exception as e:
        response = {"success": False, "message": f"Error: {str(e)}"}

    return jsonify(response)

@app.route('/select_target', methods=['POST'])
def select_target():
    global dataset, target_column

    target_column = request.form['target_column']

    # Perform validation if the target column exists in the dataset
    if target_column in dataset.columns:
        # Remove the target column from the dataset
        # dataset = dataset.drop(columns=[target_column])

        # Continue with the tasks or redirect to the next step
        return redirect(url_for('tasks'))
    else:
        return "Error: Invalid target column. Please choose a correct column."

@app.route('/tasks')
def tasks():
    return render_template('tasks.html')

@app.route('/perform_task', methods=['POST'])
def perform_task():
    global dataset  # Make sure to declare dataset as global
    task = request.form['task']
    
    result = None
    
    if task == 'describe':
        return redirect(url_for('describe_task'))
    elif task == 'remove_null':
        return redirect(url_for('imputation_tasks'))
    elif task == 'encode_categorical':
        return render_template('select_encoding.html', categorical_columns=dataset.select_dtypes(include=['object']).columns.tolist())
    elif task == 'scale_features':
         return render_template('select_scaling_option.html')
    elif task == 'model_build':
        return render_template('model_build_task.html')
    elif task == 'download_dataset':
        return redirect(url_for('download_dataset'))
    else:
        result = "Invalid task selected."

    return render_template('result.html', result=result)


@app.route('/describe_task')
def describe_task():
    return render_template('describe_task.html', columns=dataset.columns.tolist())

@app.route('/describe_column', methods=['POST', 'GET'])
def describe_column():
    if request.method == 'POST':
        column_name = request.form['column_name']
        # Implement code to describe the specific column
        description = dataset[column_name].describe().to_string()
        return render_template('result.html', result=description)
    else:
        return render_template('describe_column.html', columns=dataset.columns.tolist())

@app.route('/show_properties')
def show_properties():
    # Implement code to show properties of each column
    properties = dataset.describe().to_string()
    return render_template('result.html', result=properties)

@app.route('/show_dataset')
def show_dataset():
    global dataset
    return render_template('show_dataset.html', dataset=dataset.to_html())

@app.route('/download_dataset')
def download_dataset():
    try:
        # Download the modified dataset as CSV
        output = BytesIO()  # Use BytesIO instead of StringIO
        dataset.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)

        return send_file(output, as_attachment=True, download_name='modified_dataset.csv', mimetype='text/csv')

    except Exception as e:
        flash(f"Error: {str(e)}", 'error')
        return redirect(url_for('tasks'))
        
@app.route('/imputation_tasks', methods=['GET', 'POST'])
def imputation_tasks():
    global dataset  # Make sure to declare dataset as global
    columns_with_null = dataset.columns[dataset.isnull().any()].tolist()
    return render_template('imputation_tasks.html', columns_with_null=columns_with_null)

@app.route('/impute', methods=['POST'])
def impute():
    global dataset  # Make sure to declare dataset as global
    imputation_task = request.form['imputation_task']

    try:
        if imputation_task == 'show_null_values':
            null_values = 100 * dataset.isnull().sum() / len(dataset)
            result = "No null values found." if null_values.empty else null_values.to_string()
        elif imputation_task == 'remove_columns':
            columns_with_null = dataset.columns[dataset.isnull().any()].tolist()
            if not columns_with_null:
                result = "No columns with null values found."
            else:
                return render_template('imputation_tasks.html', imputation_task=imputation_task, columns_with_null=columns_with_null)
        elif imputation_task == 'show_dataset_after_imputation':
            return render_template('show_dataset.html', dataset=dataset.to_html())
        elif imputation_task in ['fill_mean', 'fill_median', 'fill_mode']:
            # Extract the imputation method from the task name
            columns_with_null = dataset.columns[dataset.isnull().any()].tolist()
            imputation_method = imputation_task.split('_')[1]
            return render_template('select_column_to_fill.html', imputation_task=imputation_method, columns_with_null=columns_with_null)
        else:
            result = "Invalid imputation task selected."

    except Exception as e:
        error_message = f"Error: {str(e)}"
        flash(error_message, 'error')
        return redirect(url_for('imputation_tasks'))

    return render_template('result.html', result=result)

@app.route('/select_column_to_fill', methods=['POST'])
def select_column_to_fill():
    global dataset  # Make sure to declare dataset as global
    imputation_task = request.form['imputation_task']
    selected_column = request.form['selected_column']
    # print(f"Imputation Task: {imputation_task}")
    # print(f"Selected Column: {selected_column}")

    try:
        if not is_numeric_dtype(dataset[selected_column]) and imputation_task in ['mean', 'median']:
            # Show an alert for invalid imputation on a categorical column
            flash(f"Invalid imputation task '{imputation_task}' for a categorical column '{selected_column}'.", 'error')
        else:
            # For numeric columns, impute with mean, median, or mode
            if imputation_task == 'mean':
                dataset[selected_column].fillna(dataset[selected_column].mean(), inplace=True)
                flash(f"Imputation task '{imputation_task}' completed for column '{selected_column}'.", 'success')
            elif imputation_task == 'median':
                dataset[selected_column].fillna(dataset[selected_column].median(), inplace=True)
                flash(f"Imputation task '{imputation_task}' completed for column '{selected_column}'.", 'success')
            elif imputation_task == 'mode':
                dataset[selected_column].fillna(dataset[selected_column].mode()[0], inplace=True)
                flash(f"Imputation task '{imputation_task}' completed for column '{selected_column}'.", 'success')
        
            else:
                flash(f"Error: Column not found in the dataset.")
        
        return redirect(url_for('imputation_tasks'))

    except Exception as e:
        error_message = f"Error: {str(e)}"
        flash(error_message, 'error')
        return redirect(url_for('imputation_tasks'))


@app.route('/show_categorical_columns')
def show_categorical_columns():
    # Implement code to show categorical columns
    categorical_columns = dataset.select_dtypes(include=['object']).columns.tolist()
    return render_template('result.html', result=categorical_columns)

@app.route('/perform_one_hot_encoding')
def perform_one_hot_encoding():
    global dataset  # Make sure to declare dataset as global
    try:
        # Assuming 'categorical_column' is the column to be encoded
        dataset = pd.get_dummies(dataset, columns=['categorical_column'])
        result = "One Hot encoding performed successfully."
    except Exception as e:
        error_message = f"Error: {str(e)}"
        flash(error_message, 'error')
        return redirect(url_for('tasks'))

    return render_template('result.html', result=result)

@app.route('/select_encoding')
def select_encoding():
    return render_template('select_encoding.html', categorical_columns=dataset.select_dtypes(include=['object']).columns.tolist())

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

@app.route('/perform_encoding_task', methods=['POST'])
def perform_encoding_task():
    global dataset  # Make sure to declare dataset as global
    encoding_option = request.form['encoding_option']

    if encoding_option != 'show_dataset':
        # Assuming 'selected_columns' is the list of columns to be encoded
        selected_columns = request.form.getlist('selected_columns')

        try:
            print(f"Encoding option: {encoding_option}")
            print(f"Selected columns: {selected_columns}")

            if dataset.empty:
                raise ValueError("Error: Dataset is empty.")

            if encoding_option == 'one_hot':
                dataset = pd.get_dummies(dataset, columns=selected_columns)
                result = "One Hot encoding performed successfully."
            elif encoding_option == 'label':
                label_encoder = LabelEncoder()
                for column in selected_columns:
                    dataset[column] = label_encoder.fit_transform(dataset[column])
                result = "Label encoding performed successfully."
            else:
                result = "Invalid encoding option selected."
            
            print("Encoded Dataset:")
            print(dataset.head())  # Check the first few rows of the encoded dataset

        except Exception as e:
            error_message = f"Error: {str(e)}"
            flash(error_message, 'error')
            return redirect(url_for('tasks'))

        return render_template('result.html', result=result)

    else:
        return render_template('show_dataset.html', dataset=dataset.to_html())
@app.route('/select_scaling_option', methods=['POST'])
def select_scaling_option():
    scaling_option = request.form['scaling_option']

    if scaling_option == 'perform_normalization':
        # Redirect to the page with options for normalization
        return redirect(url_for('select_normalization_option'))

    elif scaling_option == 'perform_standardization':
        # Redirect to the page with options for standardization
        return redirect(url_for('select_standardization_option'))

    elif scaling_option == 'show_dataset_after_scaling':
        # Redirect to the page to show the dataset after scaling
        return render_template('show_dataset.html', dataset=dataset.to_html())

    else:
        return "Invalid scaling option selected."

# Add these routes in app.py
@app.route('/select_normalization_option', methods=['GET', 'POST'])
def select_normalization_option():
    if request.method == 'POST':
        normalization_option = request.form['normalization_option']

        # Check if the normalization_option is 'normalize_specific_column'
        if normalization_option == 'normalize_specific_column':
            # Pass the columns to the template
            columns = dataset.columns.tolist()

            # If everything is correct, proceed with the normalization option
            print(f"Normalization Option: {normalization_option}")

            return render_template('select_column_to_normalize.html', columns=columns, normalization_option=normalization_option)

        elif normalization_option == 'normalize_whole_dataset':
            return redirect(url_for('perform_minmax_normalization'))
        elif normalization_option == 'show_dataset_after_normalization':
            result = dataset.to_string()
            return render_template('result.html', result=result)
        else:
            result = "Invalid normalization option selected."
            return render_template('result.html', result=result)

    return render_template('select_normalization_option.html')
@app.route('/perform_minmax_normalization', methods=['POST'])
def perform_minmax_normalization():
    global dataset
    try:
        if dataset.empty:
            raise ValueError("Error: Dataset is empty.")

        normalization_option = request.form['normalization_option']

        if normalization_option == 'normalize_whole_dataset':
            # Perform MinMax normalization on the whole dataset
            scaler = MinMaxScaler()
            dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)
            result = "MinMax Normalization performed successfully for the whole dataset."
            print("Normalized Dataset:")
            print(dataset.head())  # Check the first few rows of the normalized dataset
        elif normalization_option == 'normalize_specific_column':
            # Redirect to the page to select a column for normalization
            columns = dataset.columns.tolist()
            return render_template('select_column_to_normalize.html', columns=columns, normalization_option=normalization_option)
        elif normalization_option == 'show_dataset_after_normalization':
            return render_template('show_dataset.html', dataset=dataset.to_html())
        else:
            result = "Invalid normalization option selected."
            flash(result, 'error')  # Flash the error message
            return render_template('result.html', result=result)

    except Exception as e:
        error_message = f"Error: {str(e)}"
        flash(error_message, 'error')
        return render_template('result.html', result=error_message)

    flash(result, 'success')  # Flash the success message
    return render_template('result.html', result=result)

# Add these routes in app.py
@app.route('/select_standardization_option', methods=['GET','POST'])
def select_standardization_option():
    print(f"Method: {request.method} sso") 
    if request.method == 'POST':
        standardization_option = request.form['standardization_option']
        if standardization_option == 'standardize_specific_column':
            # Pass the columns to the template
            columns = dataset.columns.tolist()

            # If everything is correct, proceed with the normalization option
            print(f"standardization Option: {standardization_option}")

            return render_template('select_column_to_standardize.html', columns=columns, standardization_option= standardization_option)

        elif standardization_option == 'standardize_whole_dataset':
            return redirect(url_for('perform_standardization_task'))
        elif standardization_option == 'show_dataset_after_standardization':
            result = dataset.to_string()
            return render_template('result.html', result=result)
        else:
            result = "Invalid standardization option selected."
            return render_template('result.html', result=result)

    return render_template('select_standardization_option.html')

@app.route('/perform_standardization_task', methods=['POST'])
def perform_standardization_task():
    global dataset
    try:
        if dataset.empty:
            raise ValueError("Error: Dataset is empty.")

        # Retrieve the standardization_option value from the form
        standardization_option = request.form['standardization_option']

        if standardization_option == 'standardize_whole_dataset':
            # Perform Standardization on the whole dataset
            scaler = StandardScaler()
            dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)
            result = "Standardization performed successfully for the whole dataset."
            print("Standardized Dataset:")
            print(dataset.head())  # Check the first few rows of the standardized dataset
        elif standardization_option == 'standardize_specific_column':
            # Redirect to the page to select a column for standardization
            columns = dataset.columns.tolist()
            return render_template('select_column_to_standardize.html', columns=columns, standardization_option=standardization_option)
        elif standardization_option == 'show_dataset_after_standardization':
            return render_template('show_dataset.html', dataset=dataset.to_html())
        else:
            result = "Invalid standardization option selected."
            flash(result, 'error')  # Flash the error message
            return render_template('result.html', result=result)

    except Exception as e:
        error_message = f"Error: {str(e)}"
        flash(error_message, 'error')
        return render_template('result.html', result=error_message)

    flash(result, 'success')  # Flash the success message
    return render_template('result.html', result=result)


# Add these routes in app.py
@app.route('/select_column_to_standardize', methods=['GET', 'POST'])
def select_column_to_standardize():
    global dataset

    if request.method == 'POST':
        column_to_standardize = request.form['column_to_standardize']

        try:
            if dataset.empty:
                raise ValueError("Error: Dataset is empty.")

            # Check if the selected column exists in the dataset
            if column_to_standardize in dataset.columns:
                # Display options for normalization
                return render_template('select_standardization_option.html', column_to_standardize=column_to_standardize)
            else:
                result = f"Error: Column '{column_to_standardize}' not found in the dataset."
                print(result)  # Print the error message
                return render_template('result.html', result=result)

        except Exception as e:
            error_message = f"Error: {str(e)}"
            flash(error_message, 'error')
            return redirect(url_for('tasks'))

    return render_template('select_column_to_standardize.html', columns=dataset.columns.tolist())

@app.route('/select_column_to_normalize', methods=['GET', 'POST'])
def select_column_to_normalize():
    global dataset

    if request.method == 'POST':
        column_to_normalize = request.form['column_to_normalize']

        try:
            if dataset.empty:
                raise ValueError("Error: Dataset is empty.")

            # Check if the selected column exists in the dataset
            if column_to_normalize in dataset.columns:
                # Display options for normalization
                return render_template('select_normalization_option.html', column_to_normalize=column_to_normalize)
            else:
                result = f"Error: Column '{column_to_normalize}' not found in the dataset."
                print(result)  # Print the error message
                return render_template('result.html', result=result)

        except Exception as e:
            error_message = f"Error: {str(e)}"
            flash(error_message, 'error')
            return redirect(url_for('tasks'))

    return render_template('select_column_to_normalize.html', columns=dataset.columns.tolist())

@app.route('/perform_normalization', methods=['POST'])
def perform_normalization():
    global dataset  # Make sure to declare dataset as global
    column_to_normalize = request.form['column_to_normalize']
    normalization_option = request.form['normalization_option']
    print(f"Normalization Option: {normalization_option}")
    print(f"Column to Normalize: {column_to_normalize}")
    try:
        if dataset.empty:
            raise ValueError("Error: Dataset is empty.")

        if normalization_option == 'normalize_specific_column':
            # Perform MinMax normalization on the selected column
            scaler = MinMaxScaler()
            dataset[column_to_normalize] = scaler.fit_transform(dataset[[column_to_normalize]])
            result = f"MinMax Normalization performed successfully for column '{column_to_normalize}'."
        else:
            result = "Invalid normalization option selected."

        print("Normalized Dataset:")
        print(dataset.head())  # Check the first few rows of the normalized dataset

    except Exception as e:
        error_message = f"Error: {str(e)}"
        flash(error_message, 'error')
        return redirect(url_for('tasks'))

    return render_template('result.html', result=result)

@app.route('/perform_standardization', methods=['POST'])
def perform_standardization():
    global dataset  # Make sure to declare dataset as global
    column_to_standardize = request.form['column_to_standardize']
    standardization_option = request.form['standardization_option']
    print(f"Standardization Option: {standardization_option}")
    print(f"Column to Standardize: {column_to_standardize}")
    try:
        if dataset.empty:
            raise ValueError("Error: Dataset is empty.")

        if standardization_option == 'standardize_specific_column':
            # Perform Standardization on the selected column
            scaler = StandardScaler()
            dataset[column_to_standardize] = scaler.fit_transform(dataset[[column_to_standardize]])
            result = f"Standardization performed successfully for column '{column_to_standardize}'."
        else:
            result = "Invalid standardization option selected."

        print("Standardized Dataset:")
        print(dataset.head())  # Check the first few rows of the standardized dataset

    except Exception as e:
        error_message = f"Error: {str(e)}"
        flash(error_message, 'error')
        return redirect(url_for('tasks'))

    return render_template('result.html', result=result)

@app.route('/model_build_task')
def model_build_task():
    return render_template('model_build_task.html')

@app.route('/get_algorithm_options', methods=['POST'])
def get_algorithm_options():
    regression_algorithms = ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVM']
    classification_algorithms = ['Logistic Regression', 'KNN', 'Naive Bayes']
    task_type = request.form['task_type']
    
    if task_type == 'regression':
        algorithms = regression_algorithms
    elif task_type == 'classification':
        algorithms = classification_algorithms
    else:
        algorithms = []

    # Prepare HTML options
    options_html = ''.join(f'<option value="{algo}">{algo}</option>' for algo in algorithms)

    return options_html


@app.route('/get_column_options', methods=['POST'])
def get_column_options():
    task_type = request.form['task_type']
    algorithm = request.form['algorithm']

    # Placeholder for column options (replace this with your actual column retrieval logic)
    if not dataset.empty:
        column_options = list(dataset.columns)
    else:
        column_options = []

    return jsonify(column_options)

@app.route('/build_model', methods=['POST'])
def build_model():
    global target_column
    selected_columns = request.form.getlist('columns[]')
    selected_algorithm = request.form['algorithm']
    task_type = request.form['task_type']

    # Placeholder for the trained model (replace this with your actual model-building logic)
    global trained_model
    trained_model = build_model_function(selected_columns, selected_algorithm, task_type, target_column)

    # Return a success response (you can customize this response based on your needs)
    return jsonify({'message': 'Model built successfully!'})

@app.route('/build_model_function', methods=['POST'])
def build_model_function(columns, algorithm, task_type, target_column):
    global dataset
    global global_model, global_X_test, global_y_test

    # Placeholder for X and y (features and target)
    X = dataset[columns]
    y = dataset[target_column]

    # Placeholder for splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Placeholder for the model
    model = None

    # Model selection based on algorithm and task type
    if algorithm == 'Random Forest':
        if task_type == 'regression':
            model = RandomForestRegressor()
        elif task_type == 'classification':
            model = RandomForestClassifier()
    elif algorithm == 'Linear Regression' and task_type == 'regression':
        model = LinearRegression()
    elif algorithm == 'Logistic Regression' and task_type == 'classification':
        model = LogisticRegression()
    elif algorithm == 'SVM':
        if task_type == 'regression':
            model = SVR()
        elif task_type == 'classification':
            model = SVC()
    elif algorithm == 'KNN':
        if task_type == 'regression':
            model = KNeighborsRegressor()
        elif task_type == 'classification':
            model = KNeighborsClassifier()
    elif algorithm == 'Naive Bayes' and task_type == 'classification':
        model = GaussianNB()
    

    # Placeholder for training the model
    model.fit(X_train, y_train)
    global_model = model
    global_X_test = X_test
    global_y_test = y_test
    y_pred = model.predict(X_test)

    if task_type == 'regression':
        score = model.score(X_test, y_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2score = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        metrics = {'score': score, 'mse': mse, 'rmse': rmse, 'r2 score': r2score, 'mae': mae}
        plt.figure(figsize=(4, 4))
        sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.5})
        plt.title('Regression Plot')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        regression_plot_path = 'static/regression_plot.png'
        plt.savefig(regression_plot_path)
        plt.close()

        # Include the scatter plot file path in the metrics
        metrics['regression_plot_path'] = regression_plot_path

        #print(metrics)
        flash(json.dumps(metrics), 'metrics')
    elif task_type == 'classification':
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        confusion_mat = confusion_matrix(y_test, y_pred)
        f1score = f1_score(y_test,y_pred, average='weighted')
        plt.figure(figsize=(4, 4))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        confusion_matrix_plot_path = 'static/confusion_matrix_plot.png'
        plt.savefig(confusion_matrix_plot_path)
        plt.close()

        # Include the confusion matrix file path in the metrics

        metrics = {'precision': precision, 'recall': recall, 'accuracy': accuracy, 'f1 score': f1score}
        metrics['confusion_matrix_plot_path'] = confusion_matrix_plot_path
        #print(metrics) 
        flash(json.dumps(metrics), 'metrics')
    
    else:
        
        jsonify({'message': 'Invalid task type!'})
        #flash(json.dumps(metrics), 'metrics')


    # Redirect to the show_metrics route
    return redirect(url_for('show_metrics'))
@app.route('/feature_importance')
def feature_importance():
    # Retrieve the stored model and data from the session
    global global_model, global_X_test, global_y_test

    # Ensure that the model and data are available
    if global_model is None or global_X_test is None or global_y_test is None:
        return "Error: Model or data not found. Please train the model first."

    # Perform permutation importance calculation
    perm_importance = permutation_importance(global_model, global_X_test, global_y_test, n_repeats=30, random_state=42)

    # Get feature importance scores and corresponding feature names
    feature_importance = perm_importance.importances_mean
    feature_names = global_X_test.columns

    # Create a DataFrame to facilitate sorting
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

    # Sort the DataFrame by importance in descending order
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Convert the DataFrame to a list of dictionaries for rendering in HTML
    importance_list = importance_df.to_dict(orient='records')

    # Render the feature importance template with the calculated importance
    return render_template('feature_importance.html', feature_importance=importance_list)


@app.route('/show_metrics')
def show_metrics():
    # Retrieve all flashed messages with the 'metrics' category
    metrics_json_list = get_flashed_messages(category_filter='metrics')

    # Combine all flashed messages into a single JSON object
    metrics = {}
    for json_str in metrics_json_list:
        metrics.update(json.loads(json_str))
    confusion_matrix_plot_path = metrics.get('confusion_matrix_plot_path')
    confusion_matrix_plot_url = None
    if confusion_matrix_plot_path:
        confusion_matrix_plot_url = url_for('static', filename='confusion_matrix_plot.png')
    regression_plot_path = metrics.get('regression_plot_path')
    regression_plot_path_url = None
    if regression_plot_path:
        regression_plot_path_url = url_for('static', filename='regression_plot.png')

    if metrics:
        return render_template('show_metrics.html', metrics=metrics, confusion_matrix_plot_url=confusion_matrix_plot_url, regression_plot_path_url = regression_plot_path_url)
    else:
        return render_template('show_metrics.html', metrics=metrics)
    
if __name__ == '__main__':
    app.run(debug=True)
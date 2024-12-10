from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
import time
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

data = None
model = None
X_test = None
y_test = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and file.filename.endswith('.csv'):
        try:
            global data
            data = pd.read_csv(file)

            # Ensure 'class' column exists
            if 'class' not in data.columns:
                return jsonify({'error': "'class' column not found in data!"})

            # Encoding categorical columns if necessary
            le = LabelEncoder()
            for column in data.columns:
                if data[column].dtype == 'object':
                    data[column] = le.fit_transform(data[column])

            print(f"Data loaded: {data.head()}")
            return jsonify({'message': 'File uploaded successfully!'})
        except Exception as e:
            return jsonify({'error': f"Error uploading file: {str(e)}"})
    return jsonify({'error': 'Invalid file format!'})


@app.route('/train', methods=['POST'])
def train_model():
    global model, X_test, y_test
    if data is None:
        return jsonify({'error': 'No data loaded!'})

    model_type = request.json.get('model')

    if model_type == 'svm':
        model = SVC(probability=True)
    elif model_type == 'decisionTree':
        model = DecisionTreeClassifier()
    elif model_type == 'logisticRegression':
        model = LogisticRegression()
    elif model_type == 'randomForest':
        model = RandomForestClassifier()
    elif model_type == 'kNearestNeighbor':
        model = KNeighborsClassifier()
    else:
        return jsonify({'error': 'Invalid model selected!'})

    try:
        X = data.drop('class', axis=1)  # Assuming 'class' is the target variable
        y = data['class']

        # Check if there are any missing values
        if X.isnull().sum().any() or y.isnull().sum() > 0:
            return jsonify({'error': 'Data contains missing values!'})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        start_time = time.time()

        model.fit(X_train, y_train)

        end_time = time.time()
        training_time = end_time - start_time

        joblib.dump(model, 'model.pkl')

        return jsonify({'message': f'{model_type} model trained successfully!',
                        'training_time': training_time})
    except Exception as e:
        return jsonify({'error': f"Error training the model: {str(e)}"})


@app.route('/metrics', methods=['GET'])
def calculate_metrics():
    metric = request.args.get('metric')
    if model is None:
        return jsonify({'error': 'Model not trained yet!'})

    X = data.drop('class', axis=1)
    y = data['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    # Ensure we use probabilities for ROC and AUC, not just predictions
    if metric == 'roc':
        try:
            if hasattr(model, "predict_proba"):
                # Model supports probability prediction, use it for ROC curve calculation
                fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
                roc_auc = auc(fpr, tpr)
                return jsonify({'result': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'roc_auc': roc_auc}})
            else:
                return jsonify({'error': "Model does not support probability prediction!"})
        except Exception as e:
            return jsonify({'error': f"Error calculating ROC: {str(e)}"})

    elif metric == 'auc':
        try:
            if hasattr(model, "predict_proba"):
                # AUC calculation needs predicted probabilities
                fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
                auc_value = auc(fpr, tpr)
                return jsonify({'result': auc_value})
            else:
                return jsonify({'error': "Model does not support probability prediction!"})
        except Exception as e:
            return jsonify({'error': f"Error calculating AUC: {str(e)}"})

    elif metric == 'confusion_matrix':
        cm = confusion_matrix(y_test, y_pred)
        return jsonify({'result': cm.tolist()})

    elif metric == 'sensitivity':
        cm = confusion_matrix(y_test, y_pred)
        tp = cm[1][1]
        fn = cm[1][0]
        sensitivity = tp / (tp + fn)
        return jsonify({'result': sensitivity})

    elif metric == 'specificity':
        cm = confusion_matrix(y_test, y_pred)
        tn = cm[0][0]
        fp = cm[0][1]
        specificity = tn / (tn + fp)
        return jsonify({'result': specificity})

    return jsonify({'error': 'Invalid metric!'})


if __name__ == '__main__':
    app.run(debug=True)

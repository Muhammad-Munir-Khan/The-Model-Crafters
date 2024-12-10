let modelTrained = false;

// Function to handle file upload
function loadAndSaveData() {
    var fileInput = document.getElementById('fileInput');
    if (fileInput.files.length > 0) {
        var formData = new FormData();
        formData.append('file', fileInput.files[0]);

        fetch('/upload', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                document.getElementById('fileSelectedMessage').style.display = 'block';
                modelTrained = false; // Reset model status when a new file is uploaded
            } else {
                alert('Error uploading file');
            }
        })
        .catch(error => {
            console.error('Error uploading file:', error);
        });
    }
}

// Function to train the model based on user selection
function trainModel() {
    var modelSelector = document.getElementById('modelSelector');
    var selectedModel = modelSelector.value;

    if (!selectedModel) {
        alert('Please select a model!');
        return;
    }

    fetch('/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.message) {
            modelTrained = true;
            alert(`${data.message}`);
        } else {
            alert('Error training the model');
        }
    })
    .catch(error => {
        console.error('Error training model:', error);
        alert('An error occurred while training the model');
    });
}

// Function to calculate confusion matrix
function calculateConfusionMatrix() {
    if (!modelTrained) {
        alert('Model not trained yet!');
        return;
    }

    fetch('/metrics?metric=confusion_matrix')
        .then(response => response.json())
        .then(data => {
            if (data.result) {
                document.getElementById('confusionMatrixResult').innerHTML = JSON.stringify(data.result);
            } else {
                alert('Error calculating confusion matrix');
            }
        })
        .catch(error => {
            console.error('Error fetching confusion matrix:', error);
            alert('An error occurred while calculating the confusion matrix');
        });
}

// Function to calculate sensitivity
function calculateSensitivity() {
    if (!modelTrained) {
        alert('Model not trained yet!');
        return;
    }

    fetch('/metrics?metric=sensitivity')
        .then(response => response.json())
        .then(data => {
            if (data.result) {
                document.getElementById('sensitivityResult').innerHTML = `Sensitivity: ${data.result.toFixed(2)}`;
            } else {
                alert('Error calculating sensitivity');
            }
        })
        .catch(error => {
            console.error('Error fetching sensitivity:', error);
            alert('An error occurred while calculating sensitivity');
        });
}

// Function to calculate specificity
function calculateSpecificity() {
    if (!modelTrained) {
        alert('Model not trained yet!');
        return;
    }

    fetch('/metrics?metric=specificity')
        .then(response => response.json())
        .then(data => {
            if (data.result) {
                document.getElementById('specificityResult').innerHTML = `Specificity: ${data.result.toFixed(2)}`;
            } else {
                alert('Error calculating specificity');
            }
        })
        .catch(error => {
            console.error('Error fetching specificity:', error);
            alert('An error occurred while calculating specificity');
        });
}

// Function to calculate ROC curve
function calculateROC() {
    if (!modelTrained) {
        alert('Model not trained yet!');
        return;
    }

    fetch('/metrics?metric=roc')
        .then(response => response.json())
        .then(data => {
            if (data.result) {
                document.getElementById('rocResult').innerHTML = `
                    <h4>ROC Curve:</h4>
                    <pre>${JSON.stringify(data.result, null, 2)}</pre>
                `;
            } else {
                alert('Error calculating ROC');
            }
        })
        .catch(error => {
            console.error('Error fetching ROC:', error);
            alert('An error occurred while calculating ROC');
        });
}

// Function to calculate AUC
function calculateAUC() {
    if (!modelTrained) {
        alert('Model not trained yet!');
        return;
    }

    fetch('/metrics?metric=auc')
        .then(response => response.json())
        .then(data => {
            if (data.result) {
                document.getElementById('aucResult').innerHTML = `AUC: ${data.result.toFixed(2)}`;
            } else {
                alert('Error calculating AUC');
            }
        })
        .catch(error => {
            console.error('Error fetching AUC:', error);
            alert('An error occurred while calculating AUC');
        });
}

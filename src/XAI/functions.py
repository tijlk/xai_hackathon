import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def create_confusion_matrix(y_true, y_pred):
    # Assuming `y_true` contains the true labels and `y_pred` contains the predicted labels
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get the class labels
    classes = ['Class 0', 'Class 1']
    
    # Plot the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Fill the confusion matrix cells with the count values
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    return plt
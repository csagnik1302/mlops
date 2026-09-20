import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def evaluate(model, x_test, y_test):

    y_pred = model.predict(x_test)

    report_dict = classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).T
    matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

    print('Classification Report:')
    print(classification_report(y_true=y_test, y_pred=y_pred))

    print('Confusion Matrix:')
    print(matrix)

import argparse
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB         # Naive Bayes
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os
import pandas as pd
import mlflow


def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])


# Start Logging
mlflow.start_run()

# enable autologging
mlflow.sklearn.autolog()

os.makedirs("./outputs", exist_ok=True)


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")
    args = parser.parse_args()

    # paths are mounted as folder, therefore, we are selecting the file from folder
    train_df = pd.read_csv(select_first_file(args.train_data))

    # Extracting the label column
    y_train = train_df.pop("binary_score")

    # convert the dataframe values to array
    X_train = train_df.pop("CleanedSwrText")

    # paths are mounted as folder, therefore, we are selecting the file from folder
    test_df = pd.read_csv(select_first_file(args.test_data))

    # Extracting the label column
    y_test = test_df.pop("binary_score")

    # convert the dataframe values to array
    X_test = test_df.pop("CleanedSwrText")

    print(f"Training with data of shape {X_train.shape}")
    
    # Use CountVectorizer to convert reviews into matrices
    vect = CountVectorizer()
    X_train_dtm = vect.fit_transform(X_train)

    # equivalent to:
    # vect.fit(X_train) # words that are on the training set
    # X_train_dtm = vect.transform(X_train)

    # Perform the same in test
    X_test_dtm = vect.transform(X_test)

    # Use Naive Bayes to predict binary score
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred = nb.predict(X_test_dtm)
    
    # Calculate accuracy.
    print((metrics.accuracy_score(y_test, y_pred)))

    print(classification_report(y_test, y_pred))

    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=nb,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=nb,
        path=os.path.join(args.model, "trained_model"),
    )

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()

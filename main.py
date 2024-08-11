import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from pkg.data_processing import load_and_preprocess_data, balance_dataset, preprocess_text
from pkg.model import tokenize_and_pad, build_model

print(f"TensorFlow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load and preprocess data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataframe_path = os.path.join(script_dir, 'data', 'reviews_Automotive_5.json.gz')

    if not os.path.exists(dataframe_path):
        print(f"Error: File not found at {dataframe_path}")
        return

    review_df = load_and_preprocess_data(dataframe_path)

    # Balance dataset
    sample_df = balance_dataset(review_df)
    sample_df = preprocess_text(sample_df)

    # Tokenize and pad sequences
    X, tokenizer = tokenize_and_pad(sample_df['reviewText'])
    y = sample_df['positive']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train model
    max_features = 2000
    embed_dim = 64
    lstm_out = 16
    model = build_model(max_features, embed_dim, lstm_out, X.shape[1])
    print(model.summary())

    batch_size = 32
    model.fit(X_train, y_train,
              epochs=10,
              batch_size=batch_size,
              validation_data=(X_test, y_test),
              callbacks=[EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=2, verbose=2)])

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import deque

import os
import numpy as np
import pandas as pd
import settings
S = settings


class NeuralNetworkServices:
    def __init__(self, settings_=settings):
        self.S = settings_
        self.data = None
        self.model = None

    def load_data(self):
        data, column_scaler, sequence_data, x, y = {}, {}, [], [], []
        df = pd.read_csv(self.S.CSV_DATA_PATH)
        df.drop(index=[i for i in range(self.S.DATA_SIZE, len(df['close']))], inplace=True)
        if "date" not in df.columns:
            df["date"] = df.index
        for col in self.S.FEATURE_COLUMNS:
            assert col in df.columns, f"'{col}' does not exist in the dataframe."
        for column in self.S.FEATURE_COLUMNS:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        df['future'] = df['close'].shift(-self.S.LOOKUP_STEP)
        last_sequence = np.array(df[self.S.FEATURE_COLUMNS].tail(self.S.LOOKUP_STEP))
        df.dropna(inplace=True)
        sequences = deque(maxlen=self.S.N_STEPS)
        for entry, target in zip(df[self.S.FEATURE_COLUMNS + ["date"]].values, df['future'].values):
            sequences.append(entry)
            sequence_data.append([np.array(sequences), target]) if len(sequences) == self.S.N_STEPS else ...
        last_sequence = np.array(
            list([s[:len(self.S.FEATURE_COLUMNS)] for s in sequences]) + list(last_sequence)).astype(np.float32)
        for seq, target in sequence_data:
            x.append(seq), y.append(target)
        data["X_train"], data["X_test"], data["y_train"], data["y_test"] = \
            train_test_split(np.array(x), np.array(y), test_size=self.S.TEST_SIZE, shuffle=self.S.SHUFFLE)
        data["df"] = df.copy()
        data["test_df"] = data["df"].loc[data["X_test"][:, -1, -1]]
        data["test_df"] = data["test_df"][~data["test_df"].index.duplicated(keep='first')]
        data["X_train"] = data["X_train"][:, :, :len(self.S.FEATURE_COLUMNS)].astype(np.float32)
        data["X_test"] = data["X_test"][:, :, :len(self.S.FEATURE_COLUMNS)].astype(np.float32)
        data['last_sequence'] = last_sequence
        data["scaler_df"] = column_scaler
        self.data = data
        return data

    def create_model(self):
        model = Sequential()
        for n_layer in range(self.S.N_LAYERS):
            if n_layer == 0:
                model.add(
                    Bidirectional(
                        LSTM(self.S.UNITS, return_sequences=True),
                        batch_input_shape=(None, self.S.N_STEPS, len(self.S.FEATURE_COLUMNS))) if self.S.BIDIRECTIONAL
                    else LSTM(self.S.UNITS, return_sequences=True,
                              batch_input_shape=(None, self.S.N_STEPS, len(self.S.FEATURE_COLUMNS))))
            elif n_layer == self.S.N_LAYERS - 1:
                model.add(Bidirectional(LSTM(self.S.UNITS, return_sequences=False)) if self.S.BIDIRECTIONAL
                          else LSTM(self.S.UNITS, return_sequences=False))
            else:
                model.add(Bidirectional(LSTM(self.S.UNITS, return_sequences=True)) if self.S.BIDIRECTIONAL
                          else LSTM(self.S.UNITS, return_sequences=True))
            model.add(Dropout(self.S.DROPOUT))
        model.add(Dense(1, activation="linear"))
        model.compile(loss=self.S.LOSS, metrics=["mean_absolute_error"], optimizer=self.S.OPTIMIZER)
        self.model = model
        return model

    def train_model(self):
        os.mkdir("fit_results") if not os.path.isdir("fit_results") else ...
        os.mkdir("logs") if not os.path.isdir("logs") else ...
        os.makedirs(os.path.dirname("fit_results//" + self.S.MODEL_NAME + ".h5"), exist_ok=True)
        self.model.fit(
            self.data["X_train"], self.data["y_train"], batch_size=self.S.BATCH_SIZE, epochs=self.S.EPOCHS,
            validation_data=(self.data["X_test"], self.data["y_test"]), callbacks=
            [ModelCheckpoint(os.path.join("fit_results", self.S.MODEL_NAME + ".h5"),
                             save_weights_only=True, save_best_only=True, verbose=1),
             TensorBoard(log_dir=os.path.join("logs", self.S.MODEL_NAME))], verbose=1)

    def get_final_df(self):
        y_test = np.squeeze(
            self.data["scaler_df"]["close"].inverse_transform(np.expand_dims(self.data["y_test"], axis=0)))
        y_pred = np.squeeze(
            self.data["scaler_df"]["close"].inverse_transform(self.model.predict(self.data["X_test"])))
        test_df = self.data["test_df"]
        test_df[f"close_{self.S.LOOKUP_STEP}"] = y_pred
        test_df[f"true_close_{self.S.LOOKUP_STEP}"] = y_test
        test_df.sort_index(inplace=True)
        return test_df

    def predict(self):
        last_sequence = np.expand_dims(self.data["last_sequence"][-self.S.N_STEPS:], axis=0)
        prediction = self.model.predict(last_sequence)
        predicted_price = self.data["scaler_df"]["close"].inverse_transform(prediction)[0][0]
        return predicted_price

    def plot_graph(self, final_df_):
        plt.style.use('dark_background')
        plt.plot(final_df_[f'true_close_{self.S.LOOKUP_STEP}'], color="#ffffff")
        plt.plot(final_df_[f'close_{self.S.LOOKUP_STEP}'], color="#4fff92")
        plt.legend(["Actual Price", "Predicted Price"])
        plt.show()


def main():
    nns = NeuralNetworkServices(S)
    nns.create_model()
    print("\n\nМенеджер сервисов нейронной сети")
    while True:
        inp = input("\nNeuralNetwork сервисы:\n"
                    "1. [ + ] Загрузить новые данные для обучения и сохранить лучший результат обучения\n"
                    "2. [ * ] Запустить предсказание на основе обученных моделей NeuralNetwork\nВвод:").strip()
        if inp in ("1", "+"):
            nns.load_data()
            nns.train_model()
            nns.model.load_weights('fit_results//' + S.MODEL_NAME + '.h5')
            print(f"\nДанные лучшего результата обучения успешно загружены\n"
                  f"path: {'fit_results//' + S.MODEL_NAME + '.h5'}\n")
            continue
        elif inp in ("2", "*"):
            print("\nСохраненные модели NeuralNetwork:\n")
            models = []
            for n, model in enumerate(os.listdir("fit_results")):
                models.append(model)
                print(f"{n}. {model}")
            while True:
                models_numb = input("\nВыбор модели NeuralNetwork по номеру:")
                try:
                    nns.model.load_weights("fit_results//" + models[int(models_numb)])
                    break
                except (ValueError, TypeError, IndexError):
                    print("\nОшибка ввода!\n")
                    continue
            nns.load_data()
            loss, mae = nns.model.evaluate(nns.data["X_test"], nns.data["y_test"], verbose=0)
            mean_absolute_error = nns.data["scaler_df"]["close"].inverse_transform([[mae]])[0][0]
            final_df = nns.get_final_df()
            future_price = nns.predict()
            print(final_df[f'close_{S.LOOKUP_STEP}'])
            np.append(final_df[f'close_{S.LOOKUP_STEP}'], future_price)
            print(f"Future price after {S.LOOKUP_STEP} is {future_price:.2f}$")
            print(f"{S.LOSS} loss: {loss}")
            print(f"Mean Absolute Error: {round(mean_absolute_error, 2)}")
            nns.plot_graph(final_df)
        else:
            print("\nОшибка ввода!\n")
            continue


if __name__ == "__main__":
    main()

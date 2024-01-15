from typing import Union
from numpy.typing import NDArray
from dataclasses import dataclass
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR

from helpers import find_yn
from errors import display_pretty_errors, get_errors


@dataclass
class InputData:
    train_features: NDArray
    train_labels: NDArray
    test_features: NDArray
    test_labels: NDArray
    y_target_train: NDArray
    y_target_test: NDArray


"""
train_features
train_labels
test_features
y_target_train
y_target_test
"""


class InputDoublingMethod:
    def __init__(
        self,
        y_sum: float,
        N: int,
        regressor: Union[SVR, GradientBoostingRegressor, RandomForestRegressor] = None,
    ) -> None:
        self.__yn_train = None
        self.__yn_test = None
        self.__y_target_train = None
        self.__y_target_test = None

        self.regressor = regressor
        self.y_sum = y_sum
        self.N = N

    def apply(
        self,
        regressor: Union[SVR, GradientBoostingRegressor, RandomForestRegressor],
        data: InputData,
    ):
        self.__y_target_train = data.y_target_train
        self.__y_target_test = data.y_target_test

        regressor.fit(data.train_features, data.train_labels)

        train_pred_z = regressor.predict(data.train_features)

        pred_z = regressor.predict(data.test_features)

        self.__yn_train = find_yn(
            train_pred_z, self.y_sum, self.N
        )  # застосування методу подвоєних виходів
        self.__yn_test = find_yn(pred_z, self.y_sum, self.N)

        return (self.__yn_train, self.__yn_test)

    def print_errors(self):
        if (
            self.__yn_train is not None
            and self.__yn_test is not None
            and self.__y_target_test is not None
            and self.__y_target_train is not None
        ):
            print("Train erros:")
            train_errors = get_errors(self.__y_target_train, self.__yn_train)
            display_pretty_errors(train_errors)
            print("Test errors")
            test_errors = get_errors(self.__y_target_test, self.__yn_test)
            display_pretty_errors(test_errors)
        else:
            print("!!! Some of values is empty")
    
    @staticmethod
    def save_errors_to_csv(data, file_path):
        index = ['MAPE', 'RMSE', 'MAE', 'Max error', 'Median absolute error', 'Mean Squared error', 'R2']

        df = pd.DataFrame(data, index=index)
        df.to_csv(file_path, index=True,encoding='utf-8-sig')



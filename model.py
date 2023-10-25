from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torchmetrics import Accuracy
from sklearn.model_selection import ParameterGrid, KFold
from tqdm.auto import tqdm
from bayes_opt import BayesianOptimization, UtilityFunction
from skopt import BayesSearchCV
from utils import init_weights
from utils_analysis import (
    calculate_probs_from_odds,
    bet_from_highest_prob_difference,
    place_bet,
    bet_from_highest_model_prob,
)


class OddsetNet(nn.Module):
    def __init__(
        self,
        hidden_layers: list[int],
        activation: str = "leaky_relu",
        dropout_pct: float = 0,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.activation = self.activation_func_from_str(activation)
        self.dropout = nn.Dropout(dropout_pct)
        self.linears = nn.ModuleList(
            [
                init_weights(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
                for i in range(len(hidden_layers) - 1)
            ]
        )
        self.linears.append(nn.Linear(hidden_layers[-1], 3))
        self.learning_rate = learning_rate

    def forward(self, x):
        for i in range(len(self.linears) - 1):

            x = self.activation(self.linears[i](x))
            x = self.dropout(x)
        x = self.linears[-1](x)
        return x

    def activation_func_from_str(self, activation: str):
        if activation == "relu":
            return F.relu
        elif activation == "leaky_relu":
            return F.leaky_relu
        elif activation == "tanh":
            return F.tanh
        elif activation == "sigmoid":
            return F.sigmoid

    def optimizer_func_from_str(self, optimizer_name):
        if optimizer_name == "adam":
            return T.optim.Adam(self.parameters(), self.learning_rate)

    def train_model(
        self,
        epochs,
        trainloader,
        X_val,
        y_val,
        y_odds_val,
        optimizer_name: str = "adam",
        criterion=nn.CrossEntropyLoss(),
        acc=Accuracy(task="multiclass", num_classes=3),
    ):
        best_acc = 0
        best_epoch = 0
        best_model_state_dict = None
        best_loss = np.inf
        roi = -1000
        opt = self.optimizer_func_from_str(optimizer_name)
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                self.train()
                # zero the parameter gradients
                opt.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                opt.step()

                # print statistics
                running_loss += loss.item()
                # validation statistics
                self.eval()
                val_output = self(X_val)
                val_acc = acc(val_output, y_val)
                val_loss = criterion(val_output, y_val)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch
                    # roi = eval_ROI(self, X_val, y_val, y_odds_val)
                    best_model_state_dict = deepcopy(self.state_dict())
                if i % 500 == 499:
                    print(
                        f"[{epoch + 1}, {i + 1:5d}] validation acc: {val_acc * 100:.2f}&"
                    )
                    print(f"[{epoch + 1}, {i + 1:5d}] validation loss: {val_loss:.2f}&")
        print("Finished Training")
        return {
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "roi": roi,
        }, best_model_state_dict

    def cross_validation(
        self,
        epochs,
        X,
        y,
        y_odds,
        batch_size,
        k,
        optimizer_name: str = "adam",
        criterion=nn.CrossEntropyLoss(),
        acc=Accuracy(task="multiclass", num_classes=3),
        file_suffix="",
    ):
        kf = KFold(n_splits=k, shuffle=True)
        best_acc_k = np.empty(k)
        best_epochs_k = np.empty(k)
        roi_k = np.empty(k)
        iteration = 0
        initial_state_dict = deepcopy(self.state_dict())
        result_dict = dict()
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx, :], X[val_idx, :]
            y_train, y_val = y[train_idx], y[val_idx]
            y_odds_train, y_odds_val = y_odds[train_idx], y_odds[val_idx]
            X_val = T.Tensor(X_val).to(T.device("cpu"))
            y_val = T.LongTensor(y_val).to(T.device("cpu"))
            ds = StatsDataSet(X_train, y_train)
            trainloader = T.utils.data.DataLoader(ds, batch_size=batch_size)
            self.load_state_dict(initial_state_dict)
            info_dict, _ = self.train_model(
                epochs,
                trainloader,
                X_val,
                y_val,
                y_odds_val,
                optimizer_name,
                criterion,
                acc,
            )
            best_acc_k[iteration] = info_dict["best_acc"]
            best_epochs_k[iteration] = info_dict["best_epoch"]
            roi_k[iteration] = info_dict["roi"]
            iteration += 1
        result_dict["batch_size"] = batch_size
        result_dict["number_folds"] = k
        result_dict["best_acc"] = np.max(best_acc_k)
        result_dict["worst_best_acc"] = np.min(best_acc_k)
        result_dict["mean_best_acc"] = np.mean(best_acc_k)
        result_dict["std_best_acc"] = np.std(best_acc_k)
        result_dict["roi"] = np.max(roi_k)
        result_dict["std_roi"] = np.std(roi_k)
        # df = pd.DataFrame.from_dict(result_dict)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        # df.to_csv(f"analysis/cv_{file_suffix}_{k}fold_{timestamp}.csv", index=False)
        return result_dict


class StatsDataSet(T.utils.data.Dataset):
    def __init__(self, X, y):
        self.x_data = T.Tensor(X).to(T.device("cpu"))
        self.y_data = T.LongTensor(y).to(T.device("cpu"))

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()
        stats = self.x_data[idx, :]
        result = self.y_data[idx]
        return stats, result


class ModelPerformanceSaver:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, best_val_loss=float("inf")):
        self.best_val_loss = best_val_loss

    def __call__(
        self, current_val_loss, current_val_acc, epoch, model, optimizer, criterion
    ):
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            print(f"\nBest validation loss: {self.best_val_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            T.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                    "acc": current_val_acc,
                },
                "outputs/best_model.pth",
            )


def predict_lowest_odds(y_odds_val):
    def predict_lowest_odds_axis(game_odds):
        sorted_odds = np.argsort(game_odds)
        if sorted_odds[0] == 0:
            return 0
        elif sorted_odds[0] == 1:
            return 1
        elif sorted_odds[0] == 2:
            return 2

    return np.apply_along_axis(predict_lowest_odds_axis, 1, y_odds_val)


def calc_baseline_lowest_odds(y_odds_val, y_val):
    pred_from_odds = predict_lowest_odds(y_odds_val)
    return Accuracy("multiclass", num_classes=3)(
        T.tensor(pred_from_odds), T.Tensor(y_val)
    )


def run_bet_simulation(
    pred_probs,
    outcomes_true,
    odds,
    starting_bankroll=100,
    base_money_placed=1,
    money_strategy="variance_adjusted",
    betting_strategy="highest_diff",
):
    counter_bets_placed = 0
    odds_probs = calculate_probs_from_odds(odds)
    if betting_strategy == "highest_diff":
        bet_predictions = bet_from_highest_prob_difference(pred_probs, odds_probs)
    elif betting_strategy == "highest_prob":
        bet_predictions = bet_from_highest_model_prob(pred_probs)
    bankroll = starting_bankroll
    for i in range(len(bet_predictions)):
        if pred_probs[i, bet_predictions[i]] < odds_probs[i, bet_predictions[i]]:
            continue
        if money_strategy == "variance_adjusted":
            # 3* becuase otherwise we get bets <.5€ which can't be placed
            money_placed = (
                0.5
                + (
                    2
                    * odds[i, bet_predictions[i]]
                    * (1 - pred_probs[i, bet_predictions[i]])
                )
                ** -1
            )
        elif money_strategy == "fixed_bet":
            money_placed = base_money_placed
        else:
            raise ValueError("please define money strategy")
        net_gain = place_bet(
            bet_predictions[i],
            outcomes_true[i],
            odds[i, bet_predictions[i]],
            money_placed,
        )
        counter_bets_placed += 1
        bankroll += net_gain
        if bankroll < 0:
            print("bankrupt")
            break
    print("bets_placed_in_total:", counter_bets_placed)
    print(bankroll)
    print(
        "roi: ",
        np.round((bankroll - starting_bankroll) / starting_bankroll, decimals=2) * 100,
        "%",
    )
    return np.round((bankroll - starting_bankroll) / starting_bankroll, decimals=2)


def eval_ROI(model, X_test, y_test, y_odds_test):
    soft = nn.Softmax(dim=1)
    pred_probs = soft(model(X_test)).detach().numpy()
    roi = run_bet_simulation(pred_probs, y_test, y_odds_test)
    return roi


def run_hyperopt(
    param_dict, epochs, trainloader, X_val, y_val, y_odds_val, file_suffix=""
):
    results = {key: [] for key in param_dict.keys()}
    results["best_acc"] = []
    results["best_epoch"] = []
    results["roi"] = []
    all_param_combinations = ParameterGrid(param_dict)
    iteration = 0
    for param_combination in all_param_combinations:
        iteration += 1
        print(f"param_combination: {iteration}/{len(all_param_combinations)}")
        print(param_combination)
        model = OddsetNet(
            param_combination["hidden_layers"],
            param_combination["activation"],
            param_combination["dropout_pct"],
        )
        optimizer = T.optim.Adam(model.parameters(), lr=param_combination["lr"])
        results_dict, model_state_dict = train_model(
            epochs=epochs,
            model=model,
            optimizer=optimizer,
            trainloader=trainloader,
            X_val=X_val,
            y_val=y_val,
            y_odds_val=y_odds_val,
        )
        param_combination.update(results_dict)
        for key in param_combination.keys():
            results[key].append(param_combination[key])
    df = pd.DataFrame.from_dict(results)
    df.loc[:, "batch_size"] = trainloader.batch_size
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(f"analysis/hyperopt_{file_suffix}_{timestamp}.csv", index=False)


def bayesian_hyperopt(X, y, batch_size, k, model, optimizer, param_info_dict):
    pass

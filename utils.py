import pandas as pd
import torch
import unidecode
import matplotlib.pyplot as plt
from torch import nn

from config import team_name_mapping


def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('outputs/accuracy.png')

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.png')

def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, 'outputs/final_model.pth')

def seaon_start_end_in_four_digits(season_start_year):
    if isinstance(season_start_year,str):
        two_digit_start = int(season_start_year[-2:])
    elif isinstance(season_start_year,int):
        two_digit_start = int(str(season_start_year)[-2:])
    else:
        raise TypeError("Please only use str or int for season_start_year")
    two_digit_end = two_digit_start + 1
    return str(two_digit_start)+str(two_digit_end)

def slugify_columns(df):
    original_colums_dict = {
        x: unidecode.unidecode(x.lower().replace(" ", "_")) for x in df.columns
    }
    df = df.rename(columns=original_colums_dict)
    return df

def rename_teams(df):
    df.loc[:,"home_team"] = df["home_team"].apply(
        lambda x: team_name_mapping.get(x.strip(), x.strip())
    )
    df.loc[:,"away_team"] = df["away_team"].apply(
        lambda x: team_name_mapping.get(x.strip(), x.strip())
    )
    return df

def init_weights(model_layer):
    if isinstance(model_layer, nn.Linear):
        torch.nn.init.xavier_normal_(model_layer.weight)
        model_layer.bias.data.fill_(0.01)
    return model_layer

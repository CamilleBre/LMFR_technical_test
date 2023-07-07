import argparse 

from src.preprocessing import get_data
from src.training import train  

def main():
    parser = argparse.ArgumentParser(description="Models training")
    parser.add_argument(
        "--data",
        type=str,
        default="./data/data.csv",
        help="Path to the file containing the data to use",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="",
        help="Any features scaling method to apply to raw data",
    )
    parser.add_argument(
        "--use_cv",
        type=str,
        default="False",
        help="Use cross-validation",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="all",
        help="Model to train. Default= all the models defined.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        help="Metric to use for training and evaluation",
    )
    parser.add_argument(
        "--hp_gridsearch",
        type=str,
        default="False",
        help="Perform hyperparameters gridsearch optimization",
    )
    args = parser.parse_args()
    print(args)
    

    X, y = get_data(args)
    df_result = train(X, y, args)
    print(df_result)


if __name__ == "__main__":
    main()

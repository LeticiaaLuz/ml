import pandas as pd

import lps_ml.datasets.iara as iara
import lps_ml.datasets.selection as ml_sel

def main():
    """Main function for the dataset info tables."""

    collection_list = [
        iara.DC.A,
        # iara.DC.OS_CPA_IN,
        # iara.DC.OS
    ]

    for sub in collection_list:
        df = sub.to_df()
        # print(df)

        # part = df.groupby(['SHIPTYPE','DETAILED TYPE']).size().reset_index(name=str(sub))
        # print(part)

        print()
        print("#######################", sub)

        for classifier in iara.CargoShipClassifier:
            selector = classifier.as_selector()
            df2 = selector.apply(df)
            part = df2.groupby(['Target']).size().reset_index(name="Qty")

            print()
            print("###", classifier)
            print(part)


if __name__ == "__main__":
    main()

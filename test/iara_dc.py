"""
IARA Dataset Info Tables Test Program

This script generates tables describing compiled information.
"""
import pandas as pd

import lps_ml.datasets.iara as iara


def main():
    """Main function for the dataset info tables."""

    os_ship_merged = []
    collection_list = [
        iara.DC.A,
        iara.DC.B,
        iara.DC.C,
        iara.DC.D,
    ]

    for sub in collection_list:
        df = sub.to_df()
        part = df.groupby(['SHIPTYPE','DETAILED TYPE']).size().reset_index(name=str(sub))

        if not isinstance(os_ship_merged, pd.DataFrame):
            os_ship_merged = part
        else:
            os_ship_merged = pd.merge(os_ship_merged, part,
                                      on=['SHIPTYPE','DETAILED TYPE'],how='outer')

    os_ship_merged = os_ship_merged.fillna(0)
    os_ship_merged = os_ship_merged.sort_values(['SHIPTYPE','DETAILED TYPE'])
    os_ship_merged['Total'] = os_ship_merged[os_ship_merged.columns[2:]].sum(axis=1)

    keeped = os_ship_merged[os_ship_merged['Total']>=20]

    filtered = os_ship_merged[os_ship_merged['Total']<20]
    filtered = filtered.groupby('SHIPTYPE').sum()
    filtered['DETAILED TYPE'] = 'Others'
    filtered.reset_index(inplace=True)

    os_ship_detailed_type = pd.concat([keeped, filtered])
    os_ship_detailed_type.sort_values(by='DETAILED TYPE', inplace=True)
    os_ship_detailed_type.sort_values(by='SHIPTYPE', inplace=True)
    os_ship_detailed_type.reset_index(drop=True, inplace=True)
    os_ship_detailed_type.loc['Total'] = os_ship_detailed_type.sum()
    os_ship_detailed_type.loc[os_ship_detailed_type.index[-1], 'SHIPTYPE'] = 'Total'
    os_ship_detailed_type.loc[os_ship_detailed_type.index[-1], 'DETAILED TYPE'] = 'Total'
    os_ship_detailed_type[os_ship_detailed_type.columns[2:]] = \
            os_ship_detailed_type[os_ship_detailed_type.columns[2:]].astype(int)

    print('------------------------ os_ship_detailed_type ----------------------------------------')
    print(os_ship_detailed_type)


    os_ship_type = os_ship_merged.groupby('SHIPTYPE').sum()
    os_ship_type = os_ship_type.drop('DETAILED TYPE', axis=1)
    os_ship_type.loc['Total'] = os_ship_type.sum()
    os_ship_type[os_ship_type.columns] = \
            os_ship_type[os_ship_type.columns].astype(int)

    print('------------------------ os_ship_type ----------------------------------------')
    print(os_ship_type)


    os_bg = iara.DC.E.to_df()

    print('\n------------------------- os_bg -----------------------------------------')
    print(os_bg)


    glider_ship_merged = []
    collection_list = [
        iara.DC.F,
        iara.DC.G,
    ]
    for sub in collection_list:
        df = sub.to_df()
        part = df.groupby(['SHIPTYPE','DETAILED TYPE']).size().reset_index(name=str(sub))

        if not isinstance(glider_ship_merged, pd.DataFrame):
            glider_ship_merged = part
        else:
            glider_ship_merged = pd.merge(glider_ship_merged, part,
                                      on=['SHIPTYPE','DETAILED TYPE'],how='outer')

    glider_ship_merged = glider_ship_merged.fillna(0)
    glider_ship_merged = glider_ship_merged.sort_values(['SHIPTYPE','DETAILED TYPE'])
    glider_ship_merged['Total'] = glider_ship_merged[glider_ship_merged.columns[2:]].sum(axis=1)

    glider_ship_merged.sort_values(by='DETAILED TYPE', inplace=True)
    glider_ship_merged.sort_values(by='SHIPTYPE', inplace=True)
    glider_ship_merged.reset_index(drop=True, inplace=True)
    glider_ship_merged.loc['Total'] = glider_ship_merged.sum()
    glider_ship_merged.loc[glider_ship_merged.index[-1], 'SHIPTYPE'] = 'Total'
    glider_ship_merged.loc[glider_ship_merged.index[-1], 'DETAILED TYPE'] = 'Total'
    glider_ship_merged[glider_ship_merged.columns[2:]] = \
            glider_ship_merged[glider_ship_merged.columns[2:]].astype(int)

    print('-------------------- glider_ship_detailed_type ------------------------------------')
    print(glider_ship_merged)


    glider_ship_type = glider_ship_merged.groupby('SHIPTYPE').sum()
    glider_ship_type = glider_ship_type.drop('DETAILED TYPE', axis=1)
    glider_ship_type.loc['Total'] = glider_ship_type.sum()
    glider_ship_type[glider_ship_type.columns] = \
            glider_ship_type[glider_ship_type.columns].astype(int)

    print('------------------------ glider_ship_type ----------------------------------------')
    print(glider_ship_type)

if __name__ == "__main__":
    main()

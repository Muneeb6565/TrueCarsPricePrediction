import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Feature_Manipulator(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X)

        # Feature 1
        conditions = [
            (df['Mileage'] <= 10),
            (df['Mileage'] > 10) & (df['Mileage'] <= 100),
            (df['Mileage'] > 100) & (df['Mileage'] <= 10000),
            (df['Mileage'] > 10000)
        ]

        values = ['new', 'used', 'used_a_lot ', 'most_used ']
        df['condition'] = np.select(conditions, values)

        # Feature 2
        conditions = [
            (df['Year'] >= 2016),
            (df['Year'] < 2016) & (df['Year'] >= 2014),
            (df['Year'] < 2014) & (df['Year'] >= 2012),
            (df['Year'] < 2012)
        ]

        values = ['New', 'Not_so_old', 'Old', 'Very_old']
        df['Age'] = np.select(conditions, values)

        # Feature 3
        df['Engine_series'] = df['Vin'].str[7:8]

        # Feature 4
        def fun(cat):

            if cat == 'Houston':
                return 'Houston'

            elif cat == 'San Antonio':
                return 'San Antonio'

            elif cat == 'Louisville':
                return 'Louisville'

            elif cat == 'Jacksonville':
                return 'Jacksonville'

            elif cat == 'Austin':
                return 'Austin'

            elif cat == 'Miami':
                return 'Miami'

            elif cat == 'Orlando':
                return 'Orlando'

            elif cat == 'Raleigh':
                return 'Raleigh'

            elif cat == 'Colorado Springs':
                return 'Colorado Springs'

            elif cat == 'Philadelphia':
                return 'Philadelphia'

            elif cat == 'Columbia':
                return 'Columbia'

            elif cat == 'Dallas':
                return 'Dallas'

            elif cat == 'Tucson':
                return 'Tucson'

            elif cat == 'Phoenix':
                return 'Phoenix'

            else:
                return "Other"

        df['change_city'] = df.City.apply(fun)

        # Feature 5
        def fun2(cat):
            if cat == 'Silverado':
                return 'Silverado'

            elif cat == 'Grand':
                return 'Grand'

            elif cat == 'Accord':
                return 'Accord'

            elif cat == 'Altima2.5':
                return 'Altima2.5'

            elif cat == 'Sierra':
                return 'Sierra'

            elif cat == 'F-1504WD':
                return 'F-1504WD'

            elif cat == 'Civic':
                return 'Civic'

            elif cat == '3':
                return '3'

            elif cat == 'Wrangler':
                return 'Wrangler'

            elif cat == 'Jetta':
                return 'Jetta'

            elif cat == 'Santa':
                return 'Santa'

            elif cat == 'FusionSE':
                return 'FusionSE'

            elif cat == 'Super':
                return 'Super'

            elif cat == 'EquinoxFWD':
                return 'EquinoxFWD'

            else:
                return "Other"

        df['changed_model'] = df.Model.apply(fun2)

        df.drop(['City', 'Year', 'Model', 'Vin'], axis=1, inplace=True)

        print('Feature Generator')
        return pd.DataFrame(df)


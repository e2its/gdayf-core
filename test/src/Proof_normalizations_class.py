
from os import path
import os
import pandas
import numpy as np
from pprint import pprint
from gdayf.normalizer.normalizer import Normalizer

if __name__ == '__main__':
    path_file = r'D:\Data\datasheets\regression\DM-Metric'
    os.chdir(path_file)
    aggregated_file = pandas.read_csv(path.join(str(path_file), 'DM-Metric.csv'))

    aggregated_file.loc[np.random.randint(0,aggregated_file.shape[0],25),'Temperature_Habitacion_Sensor'] = np.nan
    aggregated_file.loc[np.random.randint(0,aggregated_file.shape[0],25), 'Weather_Temperature'] = np.nan
    aggregated_file.loc[np.random.randint(0,aggregated_file.shape[0],25), 'Lighting_Comedor_Sensor'] = np.nan
    print(aggregated_file.shape[0])
    col = 'Temperature_Habitacion_Sensor'
    objective_col = 'Weather_Temperature'
    dataframe = aggregated_file
    pprint(dataframe)

    norm = Normalizer()
    print(dataframe.dtypes)

    dataframe.loc[:,'Date'] = norm.normalizeBase(dataframe.loc[:,'Date'])
    print(dataframe.dtypes)


    #working range
    dataframe.loc[:, 'CO2_Comedor_Sensor'] = norm.normalizeWorkingRange(dataframe.loc[:,'CO2_Comedor_Sensor'],0,100)
    print(dataframe.loc[:,'CO2_Comedor_Sensor'].max())
    print(dataframe.loc[:,'CO2_Comedor_Sensor'].min())

    #StdMean
    dataframe.loc[:, 'CO2_Comedor_Sensor'] = norm.normalizeStdMean(dataframe.loc[:,'CO2_Comedor_Sensor'])
    print(dataframe.loc[:, 'CO2_Comedor_Sensor'].std() )
    print(dataframe.shape[0])

    # DropMissing
    dataframe = norm.normalizeDropMissing(dataframe,objective_col)
    print(dataframe.shape[0])

    #Discretize
    dataframe.loc[:, 'CO2_Habitacion_Sensor'] = norm.normalizeDiscretize(dataframe.loc[:,'CO2_Habitacion_Sensor'], buckets_number=10, fixed_size=True)
    pprint(dataframe.loc[:,'CO2_Habitacion_Sensor'].value_counts())

    dataframe.loc[:, 'Humedad_Habitacion_Sensor'] = norm.normalizeDiscretize(dataframe.loc[:,'Humedad_Habitacion_Sensor'], buckets_number=10, fixed_size=False)
    pprint(dataframe.loc[:,'Humedad_Habitacion_Sensor'].value_counts())

    #Fixed missing
    dataframe.loc[:, 'Lighting_Comedor_Sensor'] = norm.fixedMissingValues(dataframe.loc[:, 'Lighting_Comedor_Sensor'],0.0)
    pprint(dataframe[dataframe.loc[:,'Lighting_Comedor_Sensor'].isin([0.0]) == True])
    pprint(dataframe[dataframe.loc[:,'Lighting_Comedor_Sensor'].isnull() == True])

    #Drop NaN on Objective_column
    dataframe = norm.normalizeDropMissing(dataframe, objective_col)
    pprint(dataframe[dataframe.loc[:, objective_col].isnull() == True])

    # meanMissingValues
    dataframe = norm.meanMissingValues(dataframe,col,objective_col)
    pprint(dataframe[dataframe.loc[:, col].isnull() == True])

    # progressiveMissingValues
    dataframe = norm.progressiveMissingValues(dataframe,col,objective_col)
    pprint(dataframe[dataframe.loc[:, col].isnull() == True])

    #normalizeAgregation
    dataframe.loc[:, 'Temperature_Exterior_Sensor'] = norm.normalizeAgregation(dataframe.loc[:,'Temperature_Exterior_Sensor'], 0.50)
    print(dataframe.loc[:, 'Temperature_Exterior_Sensor'].value_counts())
    print(dataframe.loc[:, 'Temperature_Exterior_Sensor'].shape[0])

    pprint(dataframe)

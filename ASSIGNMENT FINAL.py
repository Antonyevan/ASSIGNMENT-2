# -*- coding: utf-8 -*-
"""
World Bank Data Analysis Program

This program reads and analyzes World Bank data, generating visualizations such as heatmaps,
histograms, and line plots. It also calculates skewness, kurtosis, and descriptive statistics
for selected indicators. The data is filtered based on specified countries, years, and indicators.

Author: Antony Evan Alosius
Date: 13/12/23
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from stats import skew, kurtosis


def read_and_clean_world_bank_data(file_path, selected_countries, selected_years, \
                                   selected_indicators):
    """
   Read and clean World Bank data based on specified countries, years, and indicators.

   Parameters:
   - file_path: str
       Path to the World Bank data file (CSV format).
   - selected_countries: list
       List of country names to include in the analysis.
   - selected_years: list
       List of years to include in the analysis.
   - selected_indicators: list
       List of indicator codes to include in the analysis.

   Returns:
   - selected_data: DataFrame
       Cleaned DataFrame containing selected data.
   - df_transposed: DataFrame
       Transposed DataFrame with countries as columns.
   """
    # Read the World Bank data into a DataFrame
    df = pd.read_csv(file_path, skiprows=4)
    
    # Select relevant data based on specified countries, years, and indicators
    selected_data = df[df['Country Name'].isin(selected_countries)]\
        [['Country Name', 'Indicator Name', 'Indicator Code'] + selected_years]
    selected_data = \
        selected_data[selected_data['Indicator Code'].isin(selected_indicators)]
    
    # Check if the specified columns exist in the DataFrame
    required_columns = ['Country Name', 'Indicator Name', 'Indicator Code']
    if all(column in selected_data.columns for column in required_columns):
        # Set the index for the selected data
        selected_data.set_index(required_columns, inplace=True)
        
        # Transpose the DataFrame to have countries as columns
        df_transposed = selected_data.transpose()

        # Drop rows with missing values
        df_transposed = df_transposed.dropna()
        
        # Return 2 dataframes, one with year as column and one with countries as column
        return selected_data, df_transposed
    else:
        print(f"Error: Columns {required_columns} not found in the DataFrame.")


def create_heatmap(data, countries, indicators, country_name, indicator_names):
    """
    Create and display a correlation heatmap for selected indicators of a specific country.

    Parameters:
    - data: DataFrame
        DataFrame containing selected data.
    - countries: list
        List of country names.
    - indicators: list
        List of indicator codes.
    - country_name: str
        Name of the specific country for which the heatmap is created.
    - indicator_names: list
        List of indicator names.

    Returns:
    None
    """
    correlation_matrix = data.corr()
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, cmap='viridis', annot=True, fmt=".2f", \
                cbar_kws={'label': 'Correlation'}, linewidths=.5)
    plt.title(f"Correlation Heatmap for "+country_name)
    plt.xlabel('indicators')
    plt.ylabel('indicators')
    plt.yticks([i + 0.5 for i in range(len(indicator_names))], indicator_names)
    plt.xticks([i + 0.5 for i in range(len(indicator_names))], indicator_names)
    # Show the plot
    plt.show()

    
def create_histogram(selected_data, selected_years):
    """
   Create and display side-by-side bar plots for selected indicators over specified years.

   Parameters:
   - selected_data: DataFrame
       DataFrame containing selected data.
   - selected_years: list
       List of years.

   Returns:
   None
   """
    # Reset the index to make 'Country Name', 'Indicator Name', and 'Indicator Code' regular columns
    selected_data_reset = selected_data.reset_index()

    urban_population = selected_data_reset[selected_data_reset\
                                           ['Indicator Code'] == 'SP.URB.TOTL']
    AGF_GDP = selected_data_reset[selected_data_reset\
                                  ['Indicator Code'] == 'NV.AGR.TOTL.ZS']
    
    plt.figure(figsize=(10, 9))

    # Urban population growth
    
    sns.barplot(x='Country Name', hue='variable', y='value', \
                data=pd.melt(urban_population, id_vars=['Country Name'],\
                             value_vars=selected_years))
    plt.xlabel('Country')
    plt.ylabel('Urban population')
    plt.title('Urban population (2015-2020)')
    plt.legend(title='Year')

    # Agriculture, forestry, and fishing, value added (% of GDP)
    plt.figure(figsize=(10, 9))

    sns.barplot(x='Country Name', hue='variable', y='value', data=pd.melt\
                (AGF_GDP, id_vars=['Country Name'], value_vars=selected_years))
    plt.xlabel('Country')
    plt.ylabel('Agriculture, forestry, and fishing, value added (% of GDP)')
    plt.title('Agriculture, forestry, and fishing,\
              value added (% of GDP)- 2015-2020')
    plt.legend(title='Year')

    plt.tight_layout()
    plt.show()
    
    
def create_lineplot(selected_data, selected_years):
    """
    Create and display line plots for selected indicators over specified years.

    Parameters:
    - selected_data: DataFrame
        DataFrame containing selected data.
    - selected_years: list
        List of years.

    Returns:
    None
    """
    selected_data_reset = selected_data.reset_index()
    foreign_investment = selected_data_reset[selected_data_reset\
                                             ['Indicator Code'] ==\
                                                 'BX.KLT.DINV.WD.GD.ZS']
    mortality_rate = selected_data_reset[selected_data_reset\
                                         ['Indicator Code'] == \
                                             'EG.FEC.RNEW.ZS']
    
    plt.figure(figsize=(10, 9))
    
    for country in foreign_investment ['Country Name'].unique():
        country_data = foreign_investment[foreign_investment\
                                          ['Country Name'] == country]
        plt.plot(selected_years, country_data\
                 [selected_years].values.flatten(),\
                     label=country, linestyle='--')

    plt.xlabel('Year')
    plt.ylabel('Total greenhouse gas emissions (kt of CO2 equivalent)')
    plt.title('Foreign investment')
    plt.legend(title='Country')
    plt.grid(True)

    plt.figure(figsize=(10, 9))

    for country in mortality_rate['Country Name'].unique():
        country_data =mortality_rate[mortality_rate['Country Name'] == country]
        plt.plot(selected_years, country_data\
                 [selected_years].values.flatten(), label=country, linestyle='--')

    plt.xlabel('Year')
    plt.ylabel('Renewable energy consumption (% of total final energy consumption)')
    plt.title('Renewable energy consumption')
    plt.legend(title='Country')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
    
def calculate_skewness_and_kurtosis(data):
    """
   Calculate and print skewness and kurtosis for selected indicators of a country.

   Parameters:
   - country_values_df: DataFrame
       DataFrame containing values for a specific country.
   - selected_indicators: list
       List of selected indicator codes.

   Returns:
   None
   """
    skewness_results = []
    kurtosis_results = []

    for column in data.columns:
        # Skip columns that are not numeric
        if pd.api.types.is_numeric_dtype(data[column]):
            # Calculate skewness and kurtosis using the provided functions
            column_data = data[column].dropna()
            skewness = skew(column_data)
            kurt = kurtosis(column_data)

            # Append results to the lists
            skewness_results.append((column, skewness))
            kurtosis_results.append((column, kurt))

            print(f"Column: {column}")
            print(f"Skewness: {skewness}")
            print(f"Kurtosis: {kurt}\n")

    return skewness_results, kurtosis_results


def describe_dataframe(data):
    """
    Applies the describe method to a DataFrame and prints each statistical value.

    Parameters:
    - data: DataFrame
        The input DataFrame.

    Returns:
    None
    """
    # Apply the describe method to the DataFrame
    description = data.describe()

    # Print each statistical value
    for column in description.columns:
        print(f"Column: {column}")
        print(description[column])
        print("\n")
        

def main():
    """
    Main function to execute the analysis.

    Returns:
    None
    """
    file_path = 'WB.csv'

    # Define selected parameters
    selected_countries = ['Pakistan', 'New Zealand', 'India', 'United States','Sweden']
    selected_years = [str(year) for year in range(2015, 2021)]
    selected_indicators = \
        ['SP.URB.TOTL','SH.DYN.MORT','BX.KLT.DINV.WD.GD.ZS',\
         'SE.ENR.PRSC.FM.ZS','EG.FEC.RNEW.ZS','NV.AGR.TOTL.ZS',\
             'EN.ATM.GHGT.KT.CE']

    #Call funtion to read and clean dataframe
    selected_data, df_transposed = \
        read_and_clean_world_bank_data(file_path, selected_countries, selected_years,\
                                       selected_indicators)
    df_countries = df_transposed.copy()
    
    #Specify the required country name to plot heatmap
    country_name = 'Sweden'
    
    #Fetch data to the corresponding country
    country_values_df = df_countries[country_name]
    
    #Fetch corresponding indicator names
    indicators1 = country_values_df.columns.tolist()
    indicator_names = [t[0] for t in indicators1]
    static_indicators =\
        ['Mortality rate','Nitrous oxide emission','CO2 emissions',\
         'Renewable energy consumption',\
             'Electricity prod. from renewable resources','Foreign investment',\
                 'Forest area']
    
    #Funtion call for heatmap
    create_heatmap(country_values_df, selected_countries,\
                   selected_indicators,country_name,indicator_names)
    
    #Function call for histogram
    create_histogram(selected_data, selected_years)
    
    #Function call for lineplot
    create_lineplot(selected_data, selected_years)
    
    #Function call to find skewness and Kurtosis
    skewness_results, kurtosis_results \
        = calculate_skewness_and_kurtosis(country_values_df)
    
    #Function call to describe
    describe_dataframe(country_values_df)


if __name__ == "__main__":
    main()

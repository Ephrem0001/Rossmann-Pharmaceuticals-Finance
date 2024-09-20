import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import holidays
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import ttest_ind
import os
import logging

# Define the log directory
log_dir = r'c:\Users\ephre\Documents\Rossmann-Pharmaceuticals-Finance-1\Logs'

# Create the log directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_dir, 'combined.log'),
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def visualize_promo_interval_distribution(df_Train, df_Test, promo_column='Promo', title='PromoInterval Distribution: Train vs Test'):
    logger.info("Starting visualization of promo interval distribution...")
    
    try:
        plt.figure(figsize=(10, 6))
        # Plot for training set
        sns.countplot(x=promo_column, data=df_Train, color='blue', alpha=0.6, label='Train')
        logger.info(f"Plotted {promo_column} for the training dataset.")
    
        # Plot for test set
        sns.countplot(x=promo_column, data=df_Test, color='orange', alpha=0.6, label='Test')
        logger.info(f"Plotted {promo_column} for the test dataset.")
    
        # Set plot labels and title
        plt.title(title)
        plt.xlabel(promo_column)
        plt.ylabel('Count')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Display the plot
        plt.show()
        logger.info("Visualization completed successfully.")
    
    except Exception as e:
        logger.error(f"Error occurred during visualization: {e}")


def chi_square_test(df_train, df_test, promo_column='Promo'):
    logger.info("Starting Chi-Square test.")
    try:
        # Create a contingency table for the specified promo column in both datasets
        contingency_table = pd.crosstab(df_train[promo_column], df_test[promo_column])
        logger.info(f"Contingency table created:\n{contingency_table}")

        # Perform the Chi-Square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        logger.info(f"Chi-Square test completed: chi2={chi2}, p-value={p_value}, dof={dof}")

        # Output the results
        logger.info(f'Chi-Square Test Statistic: {chi2}')
        logger.info(f'p-value: {p_value}')
        logger.info("-------------------------------------------------------")

        if p_value < 0.05:
            logger.info("The distributions are significantly different.")
        else:
            logger.info("The distributions are not significantly different.")

    except Exception as e:
        logger.error(f"Error occurred during Chi-Square test: {e}")


def categorize_and_plot_holiday_sales(df_train, holiday_col, days_before=3, days_after=3):
    logger.info("Starting categorization and plotting for holiday sales behavior.")
    try:
        # Sort by Date
        df_train.sort_values(by='Date', inplace=True)
        logger.info("Data sorted by date.")

        # Create a function to categorize periods around holidays
        def categorize_period(row, holiday_col):
            if row[holiday_col] == '0':
                return 'Normal'
            else:
                holiday_date = row['Date']
                before = df_train['Date'] >= (holiday_date - pd.DateOffset(days=days_before))
                after = df_train['Date'] <= (holiday_date + pd.DateOffset(days=days_after))

                # Determine period
                if before.any():
                    return 'Before Holiday'
                elif after.any():
                    return 'After Holiday'
                else:
                    return 'During Holiday'

        # Apply categorization
        df_train['Period_' + holiday_col] = df_train.apply(lambda row: categorize_period(row, holiday_col), axis=1)
        logger.info(f"Categorization applied to {holiday_col}.")

        # Aggregate sales by holiday periods
        holiday_sales = df_train.groupby('Period_' + holiday_col)['Sales'].mean()
        logger.info(f"Aggregated sales by holiday periods:\n{holiday_sales}")

        # Plot the results
        plt.figure(figsize=(7, 5))
        holiday_sales.plot(kind='bar', color=['blue', 'red', 'green'], alpha=0.7)
        plt.title(f'Sales Behavior Around {holiday_col} Periods')
        plt.xlabel('Period')
        plt.ylabel('Average Sales')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        logger.info(f"Plotting completed for {holiday_col}.")

    except Exception as e:
        logger.error(f"Error occurred during categorization or plotting: {e}")


# Function to initialize holiday data and add a holiday column
def add_holiday_column(df_train, country='US'):
    logger.info(f"Starting to add holiday column for {country}.")
    try:
        if country == 'US':
            holiday_list = holidays.US(years=df_train.index.year.unique())
            logger.info(f"US holidays for the years {df_train.index.year.unique()} obtained.")
        else:
            raise ValueError(f"Holidays for {country} not implemented yet.")

        df_train['IsHoliday'] = df_train.index.isin(holiday_list)
        logger.info(f"Holiday column added to dataframe.")
        return df_train
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
    except Exception as e:
        logger.error(f"An error occurred while adding the holiday column: {e}")


# Function for time series decomposition
def decompose_time_series(df_train, column='Sales', model='additive', period=365):
    logger.info(f"Starting time series decomposition on {column}.")
    try:
        decomposition = seasonal_decompose(df_train[column], model=model, period=period)
        df_train['Trend'] = decomposition.trend
        df_train['Seasonal'] = decomposition.seasonal
        logger.info(f"Decomposition completed for {column}. Trend and Seasonal components added to dataframe.")
        return decomposition
    except Exception as e:
        logger.error(f"Error occurred during time series decomposition: {e}")

# Function to plot the decomposition results
def plot_decomposition(decomposition, df_train):
    logger.info("Plotting decomposition results.")
    try:
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(df_train['Sales'], label='Original Sales')
        plt.title('Original Sales')

        plt.subplot(3, 1, 2)
        plt.plot(df_train['Trend'], label='Trend', color='orange')
        plt.title('Trend Component')

        plt.subplot(3, 1, 3)
        plt.plot(df_train['Seasonal'], label='Seasonality', color='green')
        plt.title('Seasonal Component')

        plt.tight_layout()
        plt.show()
        logger.info("Decomposition plot completed.")
    except Exception as e:
        logger.error(f"Error occurred while plotting decomposition: {e}")

# Function to plot sales with moving average
def plot_moving_average(df_train, window=30):
    logger.info(f"Plotting moving average for {window}-day window.")
    try:
        df_train['Sales_MA'] = df_train['Sales'].rolling(window=window).mean()

        plt.figure(figsize=(10, 6))
        plt.plot(df_train.index, df_train['Sales'], label='Sales', alpha=0.5)
        plt.plot(df_train.index, df_train['Sales_MA'], label=f'{window}-Day Moving Average', color='red', linewidth=2)
        plt.title('Sales Trend with Moving Average')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()
        logger.info(f"Moving average plot completed for {window}-day window.")
    except Exception as e:
        logger.error(f"Error occurred while plotting moving average: {e}")

# Function to compare holiday vs non-holiday sales trends
def compare_holiday_non_holiday_sales(df_train):
    logger.info("Comparing holiday and non-holiday sales trends.")
    try:
        holiday_sales_trend = df_train[df_train['IsHoliday']]['Sales_MA']
        non_holiday_sales_trend = df_train[~df_train['IsHoliday']]['Sales_MA']

        plt.figure(figsize=(10, 6))
        plt.plot(holiday_sales_trend.index, holiday_sales_trend, label='Holiday Sales Trend', color='blue')
        plt.plot(non_holiday_sales_trend.index, non_holiday_sales_trend, label='Non-Holiday Sales Trend', color='green', alpha=0.7)
        plt.title('Holiday vs Non-Holiday Sales Trend')
        plt.xlabel('Date')
        plt.ylabel('Sales Moving Average')
        plt.legend()
        plt.show()
        logger.info("Comparison of holiday vs non-holiday sales trends completed.")
    except Exception as e:
        logger.error(f"Error occurred while comparing holiday and non-holiday sales trends: {e}")

# Function to analyze the relationship between sales and customers
def analyze_sales_customers_relationship(df_train):
    logger.info("Analyzing relationship between sales and number of customers.")
    try:
        # Calculate the correlation
        correlation = df_train['Sales'].corr(df_train['Customers'])
        logger.info(f'Pearson correlation coefficient: {correlation}')

        # Plot the relationship
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Customers', y='Sales', data=df_train)
        plt.title('Sales vs. Number of Customers')
        plt.xlabel('Number of Customers')
        plt.ylabel('Sales')
        plt.show()
        logger.info("Sales vs Customers plot completed.")
    except Exception as e:
        logger.error(f"Error occurred during sales-customers analysis: {e}")


def analyze_promotions(df_train):
    logging.info("Analyzing promotions in the dataset")

    # Create a column indicating if the date is during a promotion
    df_train['DuringPromo'] = df_train['Promo'] == 1
    logging.info("Created 'DuringPromo' column to indicate promotion periods")

    # Compare sales during promotions vs non-promotions
    promo_sales = df_train[df_train['DuringPromo']]['Sales']
    non_promo_sales = df_train[~df_train['DuringPromo']]['Sales']
    logging.info("Separated sales data into promotion and non-promotion periods")

    # Statistical test
    t_stat, p_value = ttest_ind(promo_sales, non_promo_sales)
    logging.info(f'T-test statistic: {t_stat}, p-value: {p_value}')

    # Plot sales with and without promotions
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_train, x=df_train.index, y='Sales', hue='DuringPromo', palette={True: 'red', False: 'blue'})
    plt.title('Sales During Promotions vs Non-Promotions')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend(title='During Promo')
    plt.show()
    logging.info("Plotted sales during promotions vs non-promotions")

    # Analyze new customers acquired during promotions
    new_customers = df_train[df_train['DuringPromo']].groupby('Store').size()
    logging.info(f'Number of new customers during promotions: {len(new_customers)}')

    # Analyze repeat purchase behavior of existing customers
    existing_customers = df_train[df_train['Store'].isin(new_customers.index)]
    existing_customers_before = existing_customers[existing_customers.index < df_train[df_train['DuringPromo']].index.min()]
    existing_customers_during = existing_customers[existing_customers.index >= df_train[df_train['DuringPromo']].index.min()]
    logging.info("Analyzed repeat purchase behavior of existing customers")

    # Plot purchase frequency
    plt.figure(figsize=(12, 6))
    sns.histplot(existing_customers_before['Sales'], label='Before Promotion', color='blue', kde=True)
    sns.histplot(existing_customers_during['Sales'], label='During Promotion', color='red', kde=True)
    plt.title('Sales Frequency Before and During Promotions')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    logging.info("Plotted sales frequency before and during promotions")


def analyze_store_promotions(df_train):
    logging.info("Analyzing promotions by store")

    # Aggregate sales and promotion impact by store
    store_sales = df_train.groupby(['Store', 'Promo'])['Sales'].sum().unstack()
    store_sales.columns = ['Non-Promo', 'Promo']

    # Calculate percentage increase in sales due to promotions
    store_sales['Increase'] = (store_sales['Promo'] - store_sales['Non-Promo']) / store_sales['Non-Promo'] * 100
    logging.info("Calculated percentage increase in sales due to promotions by store")

    # Plot sales impact by store
    plt.figure(figsize=(14, 7))
    sns.barplot(x=store_sales.index, y='Increase', data=store_sales)
    plt.title('Percentage Increase in Sales Due to Promotions by Store')
    plt.xlabel('Store')
    plt.ylabel('Percentage Increase in Sales')
    plt.xticks(rotation=90)
    plt.show()
    logging.info("Plotted percentage increase in sales due to promotions by store")

    # Identify top stores for promotion deployment based on sales increase
    top_stores = store_sales.sort_values(by='Increase', ascending=False).head(10)
    logging.info(f"Top stores for promotion deployment:\n{top_stores}")


def analyze_open_store_trends(df_train):
    logging.info("Analyzing sales trends for open stores")

    # Filter out closed stores
    open_stores = df_train[df_train['Open'] == 1]
    closed_stores = df_train[df_train['Open'] == 0]
    logging.info("Filtered data for open and closed stores")

    # Analyze sales and customer trends for open stores
    open_store_sales = open_stores.groupby('Date')['Sales'].sum()
    open_store_customers = open_stores.groupby('Date')['Customers'].sum()

    # Plot Sales and Customers for open stores
    plt.figure(figsize=(14, 6))

    plt.subplot(2, 1, 1)
    plt.plot(open_store_sales, label='Sales (Open Stores)', color='blue')
    plt.title('Sales Trends During Open Store Days')
    plt.ylabel('Total Sales')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(open_store_customers, label='Customers (Open Stores)', color='green')
    plt.title('Customer Trends During Open Store Days')
    plt.ylabel('Total Customers')
    plt.legend()

    plt.tight_layout()
    plt.show()
    logging.info("Plotted sales and customer trends for open stores")


def analyze_weekday_open_stores_weekend_sales(df_train):
    logging.info("Analyzing weekend sales for stores open on weekdays")

    # Step 1: Identify stores open on all weekdays (Monday to Friday)
    weekday_data = df_train[(df_train['DayOfWeek'].isin([1, 2, 3, 4, 5])) & (df_train['Open'] == 1)]

    # Count how many weekdays each store is open
    weekday_open_count = weekday_data.groupby('Store')['DayOfWeek'].nunique().reset_index()

    # Select stores that are open on all 5 weekdays
    stores_open_all_weekdays = weekday_open_count[weekday_open_count['DayOfWeek'] == 5]['Store'].tolist()

    # Step 2: Analyze weekend sales for stores open on all weekdays
    weekend_data = df_train[df_train['DayOfWeek'].isin([6, 7])]  # Saturday = 6, Sunday = 7

    # Filter for stores that are open all weekdays
    weekend_sales_for_weekday_stores = weekend_data[weekend_data['Store'].isin(stores_open_all_weekdays)]

    # Calculate total weekend sales for these stores
    weekend_sales_summary = weekend_sales_for_weekday_stores.groupby('Store')['Sales'].sum().reset_index()

    # Log and print the results
    logging.info("Stores open on all weekdays and their weekend sales summary")
    logging.info(f"\n{weekend_sales_summary}")


def analyze_sales_customers_by_day(open_stores):
    logging.info('Starting analysis of sales and customers by day of the week.')
    
    # Group by 'DayOfWeek' and calculate mean 'Sales' and 'Customers'
    day_of_week_summary = open_stores.groupby('DayOfWeek').agg({'Sales': 'mean', 'Customers': 'mean'}).reset_index()
    logging.info('Grouped data by day of the week and calculated averages.')

    # Plot average sales and customer trends by day of the week
    plt.figure(figsize=(10, 6))
    plt.plot(day_of_week_summary['DayOfWeek'], day_of_week_summary['Sales'], 
             marker='o', label='Average Sales', color='blue')
    plt.plot(day_of_week_summary['DayOfWeek'], day_of_week_summary['Customers'], 
             marker='o', label='Average Customers', color='green')

    # Add title, labels, and legend
    plt.title('Average Sales and Customer Trends by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Average Value')
    plt.legend()

    # Set custom ticks for day labels
    plt.xticks(ticks=range(1, 8), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    # Enable grid for better readability
    plt.grid(True)

    # Show plot
    plt.tight_layout()
    plt.show()
    logging.info('Completed analysis of sales and customers by day of the week.')

def analyze_promo_impact(open_stores):
    logging.info('Starting analysis of promotional impact on sales and customers.')
    
    # Group by 'Promo' and calculate mean 'Sales' and 'Customers'
    promo_analysis = open_stores.groupby('Promo').agg({'Sales': 'mean', 'Customers': 'mean'}).reset_index()
    logging.info('Grouped data by promotion status and calculated averages.')

    # Plot Average Sales with and without Promotions
    plt.figure(figsize=(10, 6))
    plt.bar(promo_analysis['Promo'], promo_analysis['Sales'], color=['blue', 'orange'], width=0.5)
    plt.title('Average Sales with and without Promotions')
    plt.xlabel('Promotion')
    plt.ylabel('Average Sales')
    plt.xticks(ticks=[0, 1], labels=['No Promo', 'Promo'])
    plt.show()

    # Plot Average Customer Count with and without Promotions
    plt.figure(figsize=(10, 6))
    plt.bar(promo_analysis['Promo'], promo_analysis['Customers'], color=['green', 'red'], width=0.5)
    plt.title('Average Customer Count with and without Promotions')
    plt.xlabel('Promotion')
    plt.ylabel('Average Customers')
    plt.xticks(ticks=[0, 1], labels=['No Promo', 'Promo'])
    plt.show()

    logging.info('Completed analysis of promotional impact on sales and customers.')

def get_stores_open_all_weekdays(df):
    logging.info('Fetching stores that are open all weekdays.')
    
    # Define weekdays (1=Monday, 5=Friday)
    weekdays = [1, 2, 3, 4, 5]
    
    # Filter for open stores
    open_weekdays = df[df['Open'] == 1]
    
    # Count unique weekdays for each store
    stores_open_all_weekdays_group = open_weekdays[open_weekdays['DayOfWeek'].isin(weekdays)].groupby('Store')['DayOfWeek'].nunique()
    
    # Get stores that are open all weekdays
    stores_open_all_weekdays = stores_open_all_weekdays_group[stores_open_all_weekdays_group == len(weekdays)].index.tolist()
    
    logging.info(f'Found {len(stores_open_all_weekdays)} stores open all weekdays.')
    return stores_open_all_weekdays

def compare_weekend_weekday_sales(df, stores_open_all_weekdays):
    logging.info('Comparing weekend and weekday sales for stores open all weekdays.')
    
    # Weekend mapping (6=Saturday, 7=Sunday)
    weekends = [6, 7]
    
    # Filter data for weekends (Saturday and Sunday)
    weekend_sales = df[(df['DayOfWeek'].isin(weekends)) & (df['Store'].isin(stores_open_all_weekdays))]
    weekend_sales_group = weekend_sales.groupby('Store')['Sales'].mean()
    
    # Filter data for weekdays (Monday to Friday)
    weekdays = [1, 2, 3, 4, 5]
    weekday_sales = df[(df['DayOfWeek'].isin(weekdays)) & (df['Store'].isin(stores_open_all_weekdays))]
    weekday_sales_group = weekday_sales.groupby('Store')['Sales'].mean()
    
    # Combine the sales data into a single DataFrame
    sales_comparison = pd.DataFrame({'Weekday Sales': weekday_sales_group, 'Weekend Sales': weekend_sales_group})
    
    # Fill missing values with 0 for comparison
    sales_comparison.fillna(0, inplace=True)
    
    logging.info('Sales comparison between weekdays and weekends completed.')
    return sales_comparison

def plot_sales_comparison(sales_comparison):
    logging.info('Plotting sales comparison for stores open all weekdays.')
    
    # Plot sales comparison for stores open all weekdays
    sales_comparison.plot(kind='bar', figsize=(10, 6))
    plt.title('Comparison of Weekday and Weekend Sales for Stores Open on All Weekdays')
    plt.xlabel('Store')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=45)
    plt.legend(['Weekday Sales', 'Weekend Sales'])
    plt.tight_layout()
    plt.show()

    logging.info('Sales comparison plot displayed.')

def plot_average_sales_by_assortment(sales_by_assortment):
    logging.info('Starting to plot average sales by assortment type.')

    # Plot the average sales by assortment
    sales_by_assortment.plot(kind='bar', figsize=(8, 6), color='skyblue')
    plt.title('Average Sales by Assortment Type')
    plt.xlabel('Assortment Type')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    plt.show()
    
    logging.info('Average sales by assortment type plotted successfully.')

def analyze_competition_distance(df):
    logging.info('Analyzing competition distance impact on sales.')
    
    # Calculate the correlation
    correlation = df['CompetitionDistance'].corr(df['Sales'])
    logging.info(f"Correlation between CompetitionDistance and Sales: {correlation}")

    # Plot Competition Distance vs Sales
    plt.figure(figsize=(10, 6))
    plt.scatter(df['CompetitionDistance'], df['Sales'], alpha=0.3)
    plt.title('Sales vs Competition Distance')
    plt.xlabel('Competition Distance (meters)')
    plt.ylabel('Sales')
    plt.grid(True)
    plt.show()

    logging.info('Completed analysis of competition distance impact on sales.')

def analyze_city_center_sales(df):
    logging.info('Analyzing sales in city center stores.')
    
    # Filter for city center stores
    city_center_stores = df[df['StoreType'] == 'c']
    
    # Check the correlation
    correlation = city_center_stores['CompetitionDistance'].corr(city_center_stores['Sales'])
    logging.info(f"Correlation between CompetitionDistance and Sales in city centers: {correlation}")

    # Plot Sales vs Competition Distance for city center stores
    plt.figure(figsize=(10, 6))
    plt.scatter(city_center_stores['CompetitionDistance'], city_center_stores['Sales'], alpha=0.3, color='green')
    plt.title('Sales vs Competition Distance in City Centers')
    plt.xlabel('Competition Distance (meters)')
    plt.ylabel('Sales')
    plt.grid(True)
    plt.show()

    logging.info('Completed analysis of city center sales.')

def analyze_competitor_impact(df):
    logging.info('Starting analysis of competitor impact on stores.')
    
    # Check for stores with NA as CompetitionDistance and later have values
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(-1)  # Replace NA with -1 for easier analysis
    df_sorted = df.sort_values(by=['Store', 'DayOfWeek'])

    # Identify stores that transitioned from NA (-1) to a valid competitor distance
    stores_with_competitors = df_sorted.groupby('Store')['CompetitionDistance'].apply(lambda x: (x == -1).any() and (x != -1).any())
    stores_with_competitors = stores_with_competitors[stores_with_competitors].index.tolist()
    
    if stores_with_competitors:
        logging.info(f'Identified {len(stores_with_competitors)} stores that transitioned from NA to valid competitor distances.')
        
        # Filter the dataframe for these stores
        impacted_stores = df_sorted[df_sorted['Store'].isin(stores_with_competitors)]
        
        # You can add further analysis here, e.g., comparing sales before and after competitors opened
        # For demonstration, we can log a brief overview of the impacted stores
        logging.info('Overview of impacted stores:')
        logging.info(impacted_stores[['Store', 'DayOfWeek', 'Sales', 'CompetitionDistance']].head())
        
        return impacted_stores
    else:
        logging.info('No stores were found that transitioned from NA to valid competitor distances.')
        return None

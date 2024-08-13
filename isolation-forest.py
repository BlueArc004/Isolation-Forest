import csv
from datetime import datetime
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)

        num_columns = int(infile.readline().strip())

        column_names = []
        for _ in range(num_columns):
            line = infile.readline().strip()
            column_name=line.strip()
            column_names.append(column_name)
        column_names.insert(0, 'Timestamp')

        csv_writer.writerow(column_names)
        
        for line in infile:
            parts = line.split()
            if len(parts) == num_columns + 7:  # 7 additional fields for timestamp

                timestamp = datetime.strptime(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]} {parts[5]}","%Y %j %H %M %S %f")
                formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                csv_writer.writerow([formatted_timestamp] + parts[7:])

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    return df

def remove_trivial_outliers(data, alpha=0.1, beta=5.0):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        values = data[col].values
        p_low = np.percentile(values, alpha)
        p_high = np.percentile(values, 100 - alpha)
        iqr = p_high - p_low
        lower_bound = p_low - beta * iqr
        upper_bound = p_high + beta * iqr
        
        outliers = (values < lower_bound) | (values > upper_bound)
        outliers_shifted_prev = np.roll(outliers, 1)
        outliers_shifted_next = np.roll(outliers, -1)
        outliers_to_remove = outliers & ~outliers_shifted_prev & ~outliers_shifted_next
        outliers_to_remove[0] = outliers_to_remove[-1] = False
        
        if np.issubdtype(values.dtype, np.integer):
            sentinel_value = np.iinfo(values.dtype).min
            values = values.astype(float)
            values[outliers_to_remove] = sentinel_value
        else:
            values[outliers_to_remove] = np.nan
        
        data[col] = values
    
    return data

def normalize_data(data, alpha=0.1):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        p_low = np.percentile(data[col], alpha)
        p_high = np.percentile(data[col], 100 - alpha)
        data[col] = (2 / (p_high - p_low)) * (data[col] - (p_high + p_low) / 2)
    return data

def encode_categorical_columns(data):
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = pd.Categorical(data[col]).codes + 1
    return data

def preprocess_and_clean_data(df):
    df_clean = df.copy()
    df_clean = remove_trivial_outliers(df_clean)
    df_clean = normalize_data(df_clean)
    df_clean = encode_categorical_columns(df_clean)
    return df_clean

def train_isolation_forest(data, start_date, end_date, contamination=0.01):
    train_data = data.loc[start_date:end_date]
    scaler = MinMaxScaler()
    normalized_train_data = scaler.fit_transform(train_data)
    model = IsolationForest(
    n_estimators=200, 
    max_samples=1000,  
    contamination=0.001,  
    max_features=1.0, 
    bootstrap=False, 
    n_jobs=-1,  
    random_state=42,  
    verbose=1, 
    warm_start=True
    )
    model.fit(normalized_train_data)
    return model, scaler

def detect_anomalies(model, scaler, test_data):
    normalized_test_data = scaler.transform(test_data)
    test_data.loc[:, 'Anomaly'] = model.predict(normalized_test_data)
    test_data.loc[:, 'AnomalyScore'] = model.score_samples(normalized_test_data)
    return test_data

def aggregate_daily_anomalies(test_data):
    daily_anomalies = test_data.resample('D').agg({
        'Anomaly': lambda x: (x == -1).sum(),
        'AnomalyScore': 'mean'
    })
    daily_anomalies['AnomalyPercentage'] = daily_anomalies['Anomaly'] / test_data.resample('D').size() * 100
    return daily_anomalies

def print_anomaly_results(daily_anomalies, test_data):
    print(f"Number of days processed in test period: {len(daily_anomalies)}")
    print(f"Total number of data points processed in test period: {len(test_data)}")
    print("\nTop 5 Anomalous Days:")
    print(daily_anomalies.sort_values('AnomalyPercentage', ascending=False).head())

    most_anomalous_day = daily_anomalies['AnomalyPercentage'].idxmax()
    day_data = test_data.loc[most_anomalous_day.strftime('%Y-%m-%d')]
    anomalous_points = day_data[day_data['Anomaly'] == -1]
    print(f"\nDetails for most anomalous day ({most_anomalous_day.date()}):")
    print(f"Total data points: {len(day_data)}")
    print(f"Anomalous points: {len(anomalous_points)}")
    print(f"Anomaly percentage: {len(anomalous_points) / len(day_data) * 100:.2f}%")
    earliest_anomaly = anomalous_points.index.min().strftime('%H:%M:%S')
    latest_anomaly = anomalous_points.index.max().strftime('%H:%M:%S')
    print(f"Anomalies were detected from {earliest_anomaly} to {latest_anomaly}")

    

def main():
    input_file = "file"
    output_file = 'file'
    process_file(input_file, output_file)

    df = load_and_preprocess_data('file')
    df_clean = preprocess_and_clean_data(df)
    df_clean.to_csv('file')

    data = load_and_preprocess_data('file')
    print(f"The dataset is available from {data.index.min().date()} to {data.index.max().date()}")

    start_date = input("Enter the start date for training (YYYY-MM-DD): ")
    end_date = input("Enter the end date for training (YYYY-MM-DD): ")
    test_start = input("Enter the start date for testing (YYYY-MM-DD): ")
    test_end = input("Enter the end date for testing (YYYY-MM-DD): ")


    columns = [col for col in data.columns if 'Analog' in col or 'Digital' in col]
    telemetry_data = data[columns]

    # training call
    model, scaler = train_isolation_forest(telemetry_data, start_date, end_date)


    test_data = telemetry_data.loc[test_start:test_end].copy()
    test_data_with_anomalies = detect_anomalies(model, scaler, test_data)

    anomaly_count = len(test_data_with_anomalies)
    accuracy = 100 * (test_data_with_anomalies['Anomaly'] == -1).sum() / anomaly_count
    print(f"Accuracy of the model: {accuracy:.2f}%")
    

    daily_anomalies = aggregate_daily_anomalies(test_data_with_anomalies)
    print_anomaly_results(daily_anomalies, test_data_with_anomalies)

if __name__ == "__main__":
    main()

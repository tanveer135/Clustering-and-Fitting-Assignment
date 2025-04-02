""" This script performs data analysis and visualization on employee attrition data.
    It includes functions for plotting relational, categorical, and statistical plots,
    performing clustering using KMeans, and fitting a decision tree model to predict attrition.
    The script also includes functions for preprocessing the data, calculating statistical moments,
    and interpreting the results. """
# Importing necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from scipy import stats as ss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")


# Function to plot relational plot
def plot_relational_plot(df):
    """Create a scatter plot between Age and MonthlyIncome with Attrition as hue."""
    # Relational Plot: Age vs. MonthlyIncome
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Age', y='MonthlyIncome', hue='Attrition')
    plt.title("Relational Plot: Age vs Monthly Income (by Attrition)")
    plt.savefig('relational_plot.png')
    plt.close()
    # Printing the observations
    print("""Observations for Relational Plot:
        As age increases, monthly salary tends to increase,
        which aligns with the expectation that more experience
        leads to higher salary. Attrition occurs across all age
        groups, but younger individuals with lower salaries are
        more likely to leave for better job opportunities and
        salary growth.""")
    print("--------------------------------------------------------------\n")


# Function to plot categorical plot
def plot_categorical_plot(df):
    """Create a count plot for Attrition."""
    # Categorical Plot: Attrition Distribution (Bar Plot)
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Attrition', palette=['blue', 'pink'])
    plt.title("Attrition Distribution")
    plt.xlabel("Attrition (No = 0, Yes = 1)")
    plt.ylabel("Count")
    plt.savefig('categorical_plot.png')
    plt.close()
    # Printing the observations
    print("""Observations for Categorical Plot:
        The majority of employees are staying in their jobs,
        suggesting either a tendency to remain or an imbalance
        in the data, as the number of non-attriting employees
        is significantly higher than those who have attrited.""")
    print("--------------------------------------------------------------\n")


# Function to plot statistical plot
def plot_statistical_plot(df):
    """Create a boxplot for MonthlyIncome vs. Attrition."""
    # Statistical Plot: Boxplot of Monthly Income by Attrition
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df,
        x='Attrition',
        y='MonthlyIncome',
        palette=[
            'blue',
            'pink'])
    plt.title("Monthly Income Distribution by Attrition")
    plt.xlabel("Attrition (No = 0, Yes = 1)")
    plt.ylabel("Monthly Income")
    plt.savefig('statistical_plot.png')
    plt.close()
    # Printing the observations
    print("""Observations for Statistical Plot:
        The graph clearly shows that employees who attrited
        had lower monthly incomes, while those who stayed had
        comparatively higher salaries. The median salary for
        those who attrited is also quite low, indicating that
        lower salaries likely contributed to their decision to leave.""")
    print("--------------------------------------------------------------\n")


# Function to calculate statistical moments
def statistical_analysis(df, col: str):
    """Calculate statistical moments (mean, std, skewness, kurtosis) for a given column."""
    mean = df[col].mean()
    stddev = df[col].std()
    skew = df[col].skew()
    # Fisher=True for excess kurtosis
    excess_kurtosis = ss.kurtosis(df[col], fisher=True)
    return mean, stddev, skew, excess_kurtosis


# Function to write statistical moments and interpret them
def writing(moments, col):
    """Display statistical moments and interpret the skewness and kurtosis."""
    print(f"===== Statistical Analysis for '{col}' =====")
    print(f'Mean = {moments[0]:.2f}, Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and Excess Kurtosis = {moments[3]:.2f}.')

    # Interpretation
    skewness_desc = "right skewed" if moments[2] > 0 else "left skewed" if moments[2] < 0 else "not skewed"
    kurtosis_desc = "leptokurtic" if moments[3] > 0 else "platykurtic" if moments[3] < 0 else "mesokurtic"
    print("--------------------------------------------------------------")
    print(f'The data was {skewness_desc} and {kurtosis_desc}.')
    print("--------------------------------------------------------------\n")
    return


# Function to preprocess the data
def preprocessing(df):
    """Preprocess the data and calculate quick features such as 'describe', 'head/tail', and 'corr'."""
    print(df.head())
    print("--------------------------------------------------------------\n")
    print(df.tail())
    print("--------------------------------------------------------------\n")
    print(df.describe())
    print("--------------------------------------------------------------\n")
    print(df.info())
    print("---------------------------------------------------------------\n")
    print(df.corr(numeric_only=True))
    print("--------------------------------------------------------------\n")
    return df


# Function to perform clustering
def perform_clustering(df, col1, col2):
    def plot_elbow_method(scaled_data):
        inertia_values = []
        # Typically, you check for a range of clusters from 2 to 10
        k_range = range(2, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertia_values.append(kmeans.inertia_)

        # Plot Elbow Method
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertia_values, label="Inertia", marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal K')
        plt.grid(True)
        plt.savefig('elbow_plot.png')
        plt.close()

    def one_silhouette_inertia(scaled_data):
        # Perform KMeans clustering with 4 clusters
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_data)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(scaled_data, labels)
        inertia = kmeans.inertia_

        print(f"Silhouette Score: {silhouette_avg:.4f}")
        print(f"Inertia: {inertia:.4f}")
        print("--------------------------------------------------------------\n")

        return labels, inertia, kmeans.cluster_centers_

    # Gather data and scale
    data = df[[col1, col2]]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Plot elbow method to find the best number of clusters
    plot_elbow_method(scaled_data)

    # Calculate silhouette score and inertia
    labels, inertia, cluster_centers = one_silhouette_inertia(scaled_data)

    # Perform KMeans clustering with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    # Get cluster centers and transform them back to the original scale
    centroids = scaler.inverse_transform(cluster_centers)

    # Get the xkmeans and ykmeans for plotting
    # Extract the first column of the centroids (Age)
    xkmeans = centroids[:, 0]
    # Extract the second column of the centroids (Monthly Income)
    ykmeans = centroids[:, 1]

    # Return the necessary variables
    return labels, data, xkmeans, ykmeans, centroids


# Function to plot clustered data
def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1],
               c=labels, cmap='viridis', s=30)
    ax.scatter(xkmeans, ykmeans, c='red', s=100, marker='X', label='Centroids')
    ax.set_xlabel('Monthly Income')
    ax.set_ylabel('Years at Company')
    ax.set_title('KMeans Clustering: Years at Company vs Monthly Income')
    ax.legend()
    plt.savefig('clustering_plot.png')
    plt.close()
    # Printing the observations
    print("""Observations for Clustering Plot:
        We identified 4 distinct clusters based on how years at company 
        and monthly income are distributed.""")
    print("--------------------------------------------------------------\n")


# Function to perform fitting with Decision Tree model
def perform_fitting(df, col1, col2):
    """Perform fitting with a decision tree model."""
    # Gather data and prepare for fitting
    X = df[[col1]]
    # Convert to binary (0 = No, 1 = Yes)
    y = df['Attrition'].map({'Yes': 1, 'No': 0})

    # Split data into training & testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Initialize and train Decision Tree model
    model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=10,
        random_state=42)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    print("--------------------------------------------------------------\n")
    # Observations
    print("""Observation for model performance:
        Overall, the model metrics seem good, but the recall
        for attrition (class 1) is only 0.08, which suggests
        the model is not performing well at detecting attrition.
        This could be due to the data imbalance, as there are
        significantly more non-attriting employees compared to
        those who attrited.""")
    print("--------------------------------------------------------------\n")

    return X_test, y_test, y_pred  # Now returning y_pred


# Function to plot fitted data
def plot_fitted_data(X_test, y_test, y_pred):
    """Plot fitted data."""
    plt.figure(figsize=(10, 6))

    # Flattening X_test to make it a 1D array and creating the DataFrame
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        # Using .values.flatten() to convert DataFrame to numpy array
        'MonthlyIncome': X_test.values.flatten()
    })

    # Plotting
    sns.boxplot(
        x='Predicted',
        y='MonthlyIncome',
        hue='Actual',
        data=results_df,
        palette={
            0: 'blue',
            1: 'red'})

    plt.xlabel("Predicted Attrition (0 = No, 1 = Yes)")
    plt.ylabel("Monthly Income")
    plt.title("Predicted vs Actual Attrition: Monthly Income Distribution")
    plt.savefig("fitting_plot.png")
    plt.close()

    # Printing the observations
    print("""Observation for fitting plot:
        The fitting plot suggests a slight difference between
        the predicted and actual values. Additionally, employees
        who are attriting tend to have comparatively very low monthly incomes.""")
    print("--------------------------------------------------------------\n")


# Main function to execute all analysis
def main():
    # Load and preprocess the data
    df = pd.read_csv('data.csv')
    df = preprocessing(df)

    # Main column for analysis
    col = 'MonthlyIncome'

    # Plot relational, statistical, and categorical plots
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    # Perform statistical analysis and write the results
    moments = statistical_analysis(df, col)
    writing(moments, col)

    # Perform clustering
    clustering_results = perform_clustering(
        df, 'MonthlyIncome', 'YearsAtCompany')
    plot_clustered_data(*clustering_results)

    # Perform fitting
    fitting_results = perform_fitting(df, 'MonthlyIncome', 'Attrition')
    plot_fitted_data(*fitting_results)

    return


if __name__ == '__main__':
    main()
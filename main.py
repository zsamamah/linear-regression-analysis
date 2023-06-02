# Import required modules
import numpy as np
import matplotlib.pyplot as plt

# Define a function to perform linear regression analysis given two arrays of values
def linear_regression_analysis(x_values, y_values):
    # Calculate slope and intercept of linear regression line
    slope, intercept = np.polyfit(x_values, y_values, 1)
    
    # Calculate correlation coefficient and R-squared value
    corr_coef = np.corrcoef(x_values, y_values)[0, 1]
    r_squared = corr_coef ** 2
    
    # Calculate z-scores and identify outliers
    distances = abs(np.array(y_values) - (slope * np.array(x_values) + intercept))
    z_scores = np.abs((distances - np.mean(distances)) / np.std(distances, ddof=1))
    outliers = np.where(z_scores >= 2)[0]
    
    # Plot scatter plot of input data and linear regression line
    plt.scatter(x_values, y_values, color='black')
    plt.plot(x_values, slope * np.array(x_values) + intercept, color='blue')
    plt.savefig('result.png')
    
    # Highlight outliers in red and add text labels with z-scores to the plot
    for idx in outliers:
        plt.scatter(x_values[idx], y_values[idx], color='red', marker='x')
        plt.text(x_values[idx] + 0.2, y_values[idx] - 1,
                 f"Outlier Point x={x_values[idx]} y={y_values[idx]} z-score={z_scores[idx]:.2f}",
                 color='black', fontsize=8, ha='left', va='center')
        print(f"Outlier Point x={x_values[idx]} y={y_values[idx]} z-score={z_scores[idx]:.2f}")
    
    # Add title and axis labels to the plot
    plt.title('Linear Regression Analysis')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Add R-squared value to the plot as text in a box
    plt.text(0.05, 0.9, f"R-squared = {r_squared:.3f}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # Display the plot
    plt.show()
    
    # Return the R-squared value
    return r_squared

def main():
    # Read input values from file and store in arrays
    with open("Input_File.txt") as f:
        lines = f.readlines()

    x_values = []
    y_values = []
    for line in lines:
        line = line.strip().split()
        if line[0] == "X":
            x_values = [float(x) for x in line[1:]]
        elif line[0] == "Y":
            y_values = [float(y) for y in line[1:]]

    # Call the linear regression analysis function and print the R-squared value
    r_squared = linear_regression_analysis(x_values, y_values)
    print(f"R-squared = {r_squared:.3f}")

if __name__:
    main()

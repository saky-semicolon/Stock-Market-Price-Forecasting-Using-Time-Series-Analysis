"""## 2. EDA"""

data.isnull().sum()

data.describe()

# Plot the original time series
plt.figure(figsize=(12, 6))
plt.plot(data['Close'])
plt.title('Original Time Series of Stock Close Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# Plot the decomposed components
plt.figure(figsize=(12, 8))
decomposition.plot()
plt.tight_layout()
plt.show()

# Histograms of numerical features
plt.figure(figsize=(12, 6))
data.hist(bins=50, figsize=(15, 10))
plt.suptitle("Histograms of Numerical Features")
plt.tight_layout()
plt.show()

# Boxplots of numerical features to see outliers
plt.figure(figsize=(12, 6))
data.boxplot(figsize=(15, 10))
plt.suptitle("Boxplots of Numerical Features")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set the figure size globally
plt.rcParams['figure.figsize'] = [16, 6]

# Define the components and colors for individual plots
components = [
    ('Stock Price', ts_data_close, 'blue'),
    ('Trend Component', decomposition.trend, 'red'),
    ('Seasonal Component', decomposition.seasonal, 'green'),
    ('Residual Component', decomposition.resid, 'purple')
]

# Set up the date locators and formatters
year = mdates.YearLocator()
month = mdates.MonthLocator()
year_format = mdates.DateFormatter('%Y')

# Define the crash period (example, adjust if needed)
crash_start = pd.to_datetime('2020-02-20')
crash_end = pd.to_datetime('2020-04-01')

# Iterate through the components and create individual plots
for title, component, color in components:
    fig, ax = plt.subplots()
    ax.plot(component.index, component, color=color, label=title)

    # Highlight the crash period (optional)
    ax.axvspan(crash_start, crash_end, color='gray', alpha=0.3, label='Crash Period')

    ax.xaxis.set_major_locator(year)
    ax.xaxis.set_major_formatter(year_format)
    ax.xaxis.set_minor_locator(month)
    ax.set_xlabel("Date")
    ax.set_ylabel(title)
    ax.set_title(title + " Over Time")
    ax.legend()
    plt.show()

# Create the plot
fig, ax = plt.subplots(figsize=(18, 10))  # Increased figure size for presentations
ax.grid(True, linestyle='--', alpha=0.7)

# Set up the date locators and formatters
year = mdates.YearLocator()
month = mdates.MonthLocator(interval=3)
year_format = mdates.DateFormatter('%Y')

ax.xaxis.set_major_locator(year)
ax.xaxis.set_major_formatter(year_format)
ax.xaxis.set_minor_locator(month)

# Plot the data with enhanced line width for visibility
ax.plot(data.index, data['Close'], c='blue', label='Close Price', linewidth=2)
ax.plot(decomposition.trend.index, decomposition.trend, c='red', label='Trend', linewidth=3)

# Highlight the 2020 crash with clearer emphasis
crash_start = pd.to_datetime('2020-02-20')
crash_end = pd.to_datetime('2020-04-01')
ax.axvspan(crash_start, crash_end, color='red', alpha=0.3, label='2020 Market Crash')

# Add labels and title with increased size and bold formatting
ax.set_xlabel('Date', fontsize=16, fontweight='bold')
ax.set_ylabel('Price', fontsize=16, fontweight='bold')
ax.set_title('Yahoo Stock Close Price and Trend', fontsize=22, fontweight='bold')

# Add a subtitle with data date range
ax.text(0.5, 1.05, f'Data from {data.index.min().date()} to {data.index.max().date()}',
        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
        fontsize=14, style='italic')

# Enhance the legend with larger font size and bold style
ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=14, title_fontsize='13')

# Annotate significant points (max and min) with larger text, bold, and arrows
max_point = data['Close'].idxmax()
min_point = data['Close'].idxmin()

ax.annotate(f'Max: {data.loc[max_point, "Close"]:.2f}',
            xy=(max_point, data.loc[max_point, 'Close']),
            xytext=(50, 30), textcoords='offset points',
            ha='left', va='bottom',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.7', fc='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', lw=2, connectionstyle='arc3,rad=0'))

ax.annotate(f'Min: {data.loc[min_point, "Close"]:.2f}',
            xy=(min_point, data.loc[min_point, 'Close']),
            xytext=(50, -40), textcoords='offset points',
            ha='left', va='top',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.7', fc='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', lw=2, connectionstyle='arc3,rad=0'))

# Adjust layout and display
plt.tight_layout()
plt.show()

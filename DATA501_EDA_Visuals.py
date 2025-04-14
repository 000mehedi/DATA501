# Load libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro, skew, kurtosis


# import data for visuals
col_needed = [
    "Sargassum_Health_Index",
    "Marine_Biodiversity_Score",
    "Water_Temperature_C",
    "Dissolved_Oxygen_mg_L",
    "pH_Level",
    "Nutrient_Level_ppm",
]
data = pd.read_csv("UP-5_Impact_Modeling_SynData.csv", usecols=col_needed)

### bin creation for health
data["Sargassum_Health_Index"] = data["Sargassum_Health_Index"].round().astype(int)
bins = [0, 40, 70, 100]  # Upper limits for Poor, Moderate, and Good
labels = ["Poor", "Moderate", "Good"]

# Create the categorical column
data["Health_Category"] = pd.cut(
    data["Sargassum_Health_Index"], bins=bins, labels=labels, include_lowest=True
)


### Histograms #############################################################################
# Grid setup
n_cols = 3
n_rows = (len(col_needed) + n_cols - 1) // n_cols

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
axes = axes.flatten()

# Plot each histogram with Shapiro-Wilk p-value
for i, col in enumerate(col_needed):
    # Drop NaNs for the test
    col_data = data[col].dropna()

    # Shapiro-Wilk Test
    stat, p_value = shapiro(col_data)

    # Plot
    sns.histplot(
        col_data, bins=20, kde=True, ax=axes[i], color="blue", edgecolor="black"
    )

    # Format the title with p-value
    result = "Normal" if p_value > 0.05 else "Not Normal"
    title = f"{col}\nShapiro p = {p_value:.4f} ({result})"
    axes[i].set_title(title, fontsize=11)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Frequency")


# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout and main title
fig.subplots_adjust(top=0.88)
fig.suptitle("Histograms with Shapiro-Wilk Normality Test", fontsize=18, y=0.98)
plt.savefig("Histograms with Shapiro-Wilk Normality Test.png")
plt.show()

### QQ plot #############################################################################
# Grid setup
n_cols = 3
n_rows = (len(col_needed) + n_cols - 1) // n_cols

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(col_needed):
    stats.probplot(col_data, dist="norm", plot=axes[i])
    axes[i].set_title(f"Q-Q Plot: {col}", fontsize=12)

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.subplots_adjust(top=0.88)
fig.suptitle("Q-Q Plots for Environmental & Health Variables", fontsize=18, y=0.98)
plt.savefig("Q-Q Plots for Environmental & Health Variables.png")
plt.show()

### boxplots #############################################################################
# Grid setup
n_cols = 3
n_rows = (len(col_needed) + n_cols - 1) // n_cols

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(col_needed):
    sns.boxplot(x=data[col], ax=axes[i], color="blue")
    axes[i].set_title(f"Boxplot: {col}", fontsize=12)
    axes[i].set_xlabel("")

# Remove any unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout and main title
fig.subplots_adjust(top=0.88)
fig.suptitle("Boxplots for Environmental & Health Variables", fontsize=18, y=0.98)
plt.savefig("Boxplots for Environmental & Health Variables.png")
plt.show()

### C
# Select only numerical columns for correlation
numeric_data = data[
    [
        "Sargassum_Health_Index",
        "Marine_Biodiversity_Score",
        "Water_Temperature_C",
        "Dissolved_Oxygen_mg_L",
        "pH_Level",
        "Nutrient_Level_ppm",
    ]
]

# Compute correlation matrix
corr_matrix = numeric_data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 6))

# Create heatmap
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
)
plt.xticks(rotation=45, ha="right")
plt.subplots_adjust(bottom=0.25)
# Add title
plt.title("Correlation Matrix of Environmental & Health Variables", fontsize=16)
plt.savefig("Correlation Matrix of Environmental & Health Variables.png")
plt.show()

### Line Graph ###############################################################
col_needed = ["Timestamp", "Sargassum_Health_Index", "Marine_Biodiversity_Score"]
data_line = pd.read_csv("UP-5_Impact_Modeling_SynData.csv", usecols=col_needed)
data_line["Timestamp"] = pd.to_datetime(data_line["Timestamp"])
data_line.set_index("Timestamp", inplace=True)

df_line_daily = data_line.resample("2H").mean()

plt.figure(figsize=(12, 6))
sns.lineplot(df_line_daily)
plt.title("2-Hour Averages: Sargassum Health & Biodiversity")
plt.xlabel("Timestamp")
plt.ylabel("Average Score")
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig("2_Hour Averages Sargassum Health & Biodiversity.png")
plt.show()

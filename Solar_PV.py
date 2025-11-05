import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

# Load datasets
solar_data = pd.read_csv('solar_cleaned.csv')
solar_data.head(5)

print(solar_data)


solar_data.info()

# Questions
#- What do we know about solar power generation.
#- Can we see/visualize some general features about the data
#- There are 22 different PANEL_ID

# Extracting data for 2020-05-15 and plot DC produced over the whole day for all modules
#mask = (solar_data['TIME'] >= '11:00:00') & (solar_data['TIME'] <= '13:00:00')
#solar_data[mask]
plt.figure(figsize=(14, 6))
for key in solar_data['PANEL_ID'].unique():
    module_data = solar_data[(solar_data['DATE_STRING'] == '2020-05-15') & (solar_data['PANEL_ID'] == key)]
    plt.plot(module_data['TIME'], module_data['DC_POWER'], label=key)
plt.title('DC Power for All Modules on 2020-05-15')
plt.xlabel('Time')
plt.ylabel('DC Power')
plt.legend(title='PANEL_ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Select two unique PANEL_IDs
selected_keys = solar_data['PANEL_ID'].unique()[11:13]

plt.figure(figsize=(14, 6))
for key in selected_keys:
    module_data = solar_data[(solar_data['DATE_STRING'] == '2020-05-15') & (solar_data['PANEL_ID'] == key)]
    plt.plot(module_data['TIME'], module_data['DC_POWER'], label=key)
plt.title('DC Power for Two Modules on 2020-05-15')
plt.xlabel('Time')
plt.ylabel('DC Power')
plt.legend(title='PANEL_ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# What could be the reason for the difference in power production? Explore the weather data.
# Is there a correlation between weather parameters and power production?
# Can we use PCA to visualize the differences between the modules?

plt.figure(figsize=(14, 6))
for key in selected_keys:
    panel_data = solar_data[(solar_data['DATE_STRING'] == '2020-05-15') & (solar_data['PANEL_ID'] == key)]
    plt.plot(panel_data['TIME'], panel_data['IRRADIATION'], label=f'Panel {key}')
plt.title('Irradiation Over Time for Two Panels on 2020-05-15')
plt.xlabel('Time')
plt.ylabel('Irradiation')
plt.legend(title='PANEL_ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# make two plots in one figure with subplots for the irradiation plot, one for each PANEL_ID
# Export irradiation and DC power data for the selected panels to CSV
for key in selected_keys:
    panel_data = solar_data[(solar_data['DATE_STRING'] == '2020-05-15') & (solar_data['PANEL_ID'] == key)]
    export_df = panel_data[['TIME', 'DC_POWER', 'IRRADIATION']]
    export_df.to_csv(f'panel_{key}_dc_power_irradiation.csv', index=False)
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
for i, key in enumerate(selected_keys):
    panel_data = solar_data[(solar_data['DATE_STRING'] == '2020-05-15') & (solar_data['PANEL_ID'] == key)]
    axs[i].plot(panel_data['TIME'], panel_data['IRRADIATION'], label=f'Panel {key}', color='tab:blue')
    axs[i].set_title(f'Irradiation Over Time for Panel {key} on 2020-05-15')
    axs[i].set_ylabel('Irradiation')
    axs[i].legend(title='PANEL_ID', bbox_to_anchor=(1.05, 1), loc='upper left')
axs[1].set_xlabel('Time')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()  


from sklearn.decomposition import PCA

# Filter data for timepoints between 09:00:00 and 13:00:00
mask = (
    (solar_data['TIME'] >= '09:00:00') &
    (solar_data['TIME'] <= '15:00:00') &
    (solar_data['DATE_STRING'] == '2020-05-15')
)
energy_pivot = solar_data[mask].pivot_table(
    index='TIME', columns='PANEL_ID', values='DC_POWER', aggfunc='sum'
)

plt.figure(figsize=(14, 6))
sns.heatmap(energy_pivot, cmap='YlGnBu', annot=False)
plt.title('Heatmap of DC Power by PANEL_ID (11:00:00 - 13:00:00)')
plt.xlabel('PANEL_ID')
plt.ylabel('TIME')
plt.show()
energy_pivot.to_csv('dc_power_heatmap_data.csv')

# Drop rows with NaN values for PCA
energy_pivot_clean = energy_pivot.dropna()

# Transpose so that SOURCE_KEY are samples (rows), TIME are features (columns)
energy_pivot_T = energy_pivot_clean.T

# UV normalization (mean=0, std=1) of the data
energy_pivot_T_norm = (energy_pivot_T - energy_pivot_T.mean(axis=0)) / energy_pivot_T.std(axis=0)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(energy_pivot_T_norm)

# Plot PCA
plt.figure(figsize=(8, 6))
for i, label in enumerate(energy_pivot_T_norm.index):
    plt.scatter(pca_result[i, 0], pca_result[i, 1])
    plt.text(pca_result[i, 0], pca_result[i, 1], label, fontsize=9)
plt.title('PCA of DC Power (11:00:00 - 13:00:00) by SOURCE_KEY_NUMBER')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.show()
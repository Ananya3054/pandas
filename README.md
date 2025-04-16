import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi
import os

# === 1. LOAD & CLEAN DATA ===
file_path = 'GROUP-1_Description.csv'
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()
df.rename(columns={df.columns[0]: "Food Item"}, inplace=True)
df = df.loc[:, ~df.columns.str.contains('^Unnamed') | (df.columns == 'Food Item')]
df.set_index('Food Item', inplace=True)

# === 2. DEFINE NUTRIENT GROUPS ===
macronutrients = ['Caloric Value', 'Fat', 'Carbohydrates', 'Protein', 'Sugars', 
                  'Dietary Fiber', 'Cholesterol', 'Sodium', 'Water']
vitamins = [col for col in df.columns if 'Vitamin' in col]
minerals = ['Calcium', 'Copper', 'Iron', 'Magnesium', 'Manganese', 
            'Phosphorus', 'Potassium', 'Selenium', 'Zinc']
other = ['Nutrition Density']

# === 3. SAVE DIRECTORY ===
save_dir = "charts"
os.makedirs(save_dir, exist_ok=True)

# === 4. VISUALIZATION FUNCTIONS ===

def save_and_show_plot(filename):
    plt.savefig(os.path.join(save_dir, f"{filename}.png"), bbox_inches='tight')
    plt.show()

def plot_group_bar(title, columns, cmap='viridis', fname="group_bar"):
    df_subset = df[columns]
    df_subset.plot(kind='bar', figsize=(14, 6), colormap=cmap, edgecolor='black')
    plt.title(f'{title} per Food Item', fontsize=14)
    plt.ylabel('Amount per 100g')
    plt.xlabel('Food Items')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Nutrients', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_and_show_plot(fname)

def plot_top_items(nutrient, top_n=5):
    top = df[nutrient].sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top.values, y=top.index, palette='flare')
    plt.title(f'Top {top_n} Foods by {nutrient}', fontsize=13)
    plt.xlabel(f'{nutrient} (per 100g)')
    plt.ylabel('Food Item')
    for i, v in enumerate(top.values):
        plt.text(v + 0.5, i, f'{v:.1f}', color='black', va='center')
    plt.tight_layout()
    save_and_show_plot(f"top_{nutrient.replace(' ', '_')}")

def plot_radar(food_items, columns, title='Radar Chart', fname='radar_chart'):
    df_norm = df[columns].copy()
    df_norm = (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min())
    labels = columns
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    for item in food_items:
        values = df_norm.loc[item].tolist()
        values += values[:1]
        plt.polar(angles, values, label=item, linewidth=2)
    
    plt.xticks(angles[:-1], labels, color='grey', size=9)
    plt.title(title, size=13, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    save_and_show_plot(fname)

# === 5. GENERATE CHARTS ===

# Grouped Bar Charts
plot_group_bar("Macronutrients", macronutrients, cmap='coolwarm', fname='macronutrients')
plot_group_bar("Vitamins", vitamins, cmap='spring', fname='vitamins')
plot_group_bar("Minerals", minerals, cmap='summer', fname='minerals')
plot_group_bar("Nutrition Density", other, cmap='autumn', fname='nutrition_density')

# Full Nutrient Heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(df, cmap="YlGnBu", linewidths=0.5, annot=True, fmt=".1f", cbar_kws={'label': 'Nutrient Value'})
plt.title("Full Nutrient Heatmap (per 100g)", fontsize=15)
plt.xticks(rotation=45)
plt.tight_layout()
save_and_show_plot("nutrient_heatmap")

# Radar Chart
plot_radar(df.index[:3], macronutrients, title='Macronutrient Profile (Top 3 Foods)', fname='radar_macronutrients')

# Top Foods by Key Nutrients
for nutrient in ['Protein', 'Calcium', 'Iron', 'Vitamin C']:
    plot_top_items(nutrient)

# Pairplot for Macronutrients
sns.pairplot(df[macronutrients], corner=True)
plt.suptitle("Scatter Matrix of Macronutrients", y=1.02, fontsize=14)
plt.tight_layout()
save_and_show_plot("pairplot_macronutrients")

# Correlation Heatmap
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap: Nutrient Interrelationships", fontsize=14)
plt.tight_layout()
save_and_show_plot("correlation_heatmap")

# Stacked Bar Chart for Macronutrients
df_macros_percent = df[macronutrients].div(df[macronutrients].sum(axis=1), axis=0)
df_macros_percent.plot(kind='bar', stacked=True, figsize=(14, 6), colormap='tab20c', edgecolor='black')
plt.title("Macronutrient Composition (Proportional)", fontsize=14)
plt.ylabel("Proportion of Total Macronutrients")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Macronutrients', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
save_and_show_plot("stacked_macronutrients")

# Boxplot for Outliers
plt.figure(figsize=(16, 6))
sns.boxplot(data=df[macronutrients + minerals], orient="h", palette="Set2")
plt.title("Boxplot: Distribution & Outliers (Macronutrients + Minerals)", fontsize=14)
plt.xlabel("Amount per 100g")
plt.tight_layout()
save_and_show_plot("boxplot_nutrients")

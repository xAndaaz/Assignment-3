import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#dataset load
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# preprocessing
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'] = df['Title'].replace('Mme', 'Mrs')

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Custom color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
sns.set_palette(colors)

#survival rate by passenger class
ax1 = fig.add_subplot(gs[0, 0])
survival_by_class = df.groupby('Pclass')['Survived'].agg(['mean', 'count']).reset_index()
bars = ax1.bar(survival_by_class['Pclass'], survival_by_class['mean'], 
               color=colors[:3], alpha=0.8, edgecolor='black', linewidth=1)

for i, (bar, val, count) in enumerate(zip(bars, survival_by_class['mean'], survival_by_class['count'])):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.1%}\n(n={count})', ha='center', va='bottom', fontweight='bold')

ax1.set_title('Survival Rate by Passenger Class', fontweight='bold', fontsize=12)
ax1.set_xlabel('Passenger Class')
ax1.set_ylabel('Survival Rate')
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3, axis='y')

#age distribudtion
ax2 = fig.add_subplot(gs[0, 1])
survived = df[df['Survived'] == 1]['Age']
died = df[df['Survived'] == 0]['Age']

ax2.hist(died, bins=30, alpha=0.7, label='Died', color=colors[3], density=True)
ax2.hist(survived, bins=30, alpha=0.7, label='Survived', color=colors[2], density=True)

x_range = np.linspace(0, 80, 100)
kde_died = stats.gaussian_kde(died.dropna())
kde_survived = stats.gaussian_kde(survived.dropna())
ax2.plot(x_range, kde_died(x_range), color=colors[3], linewidth=2)
ax2.plot(x_range, kde_survived(x_range), color=colors[2], linewidth=2)

ax2.set_title('Age Distribution by Survival', fontweight='bold', fontsize=12)
ax2.set_xlabel('Age')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True, alpha=0.3)

#Correlation heatmap
ax3 = fig.add_subplot(gs[0, 2])
numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
corr_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, ax=ax3, cbar_kws={"shrink": .8}, fmt='.2f')
ax3.set_title('Feature Correlations', fontweight='bold', fontsize=12)

#Fare distribution
ax4 = fig.add_subplot(gs[1, 0])
df_fare = df[df['Fare'] < 300]  # Remove extreme outliers for better visualization
parts = ax4.violinplot([df_fare[df_fare['Pclass'] == i]['Fare'].dropna() for i in [1,2,3]], 
                       positions=[1,2,3], widths=0.6, showmeans=True)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)

ax4.set_title('Fare Distribution by Class', fontweight='bold', fontsize=12)
ax4.set_xlabel('Passenger Class')
ax4.set_ylabel('Fare ($)')
ax4.set_xticks([1,2,3])
ax4.grid(True, alpha=0.3)

#Survival rate by gender and embarkation
ax5 = fig.add_subplot(gs[1, 1])
survival_pivot = df.groupby(['Sex', 'Embarked'])['Survived'].mean().unstack()
survival_pivot.plot(kind='bar', ax=ax5, color=colors[1:4], alpha=0.8)
ax5.set_title('Survival Rate by Gender & Embarkation', fontweight='bold', fontsize=12)
ax5.set_xlabel('Gender')
ax5.set_ylabel('Survival Rate')
ax5.legend(title='Embarked', bbox_to_anchor=(1.05, 1), loc='upper left')
ax5.set_xticklabels(['Female', 'Male'], rotation=0)
ax5.grid(True, alpha=0.3, axis='y')

# family size impact on survival
ax6 = fig.add_subplot(gs[1, 2])
family_survival = df.groupby('FamilySize').agg({
    'Survived': ['mean', 'count']
}).round(3)
family_survival.columns = ['SurvivalRate', 'Count']
family_survival = family_survival[family_survival['Count'] >= 5]  # Filter small groups

scatter = ax6.scatter(family_survival.index, family_survival['SurvivalRate'], 
                     s=family_survival['Count']*3, alpha=0.7, color=colors[4])
ax6.plot(family_survival.index, family_survival['SurvivalRate'], 
         color=colors[4], alpha=0.5, linewidth=2)

ax6.set_title('Family Size vs Survival Rate', fontweight='bold', fontsize=12)
ax6.set_xlabel('Family Size')
ax6.set_ylabel('Survival Rate')
ax6.grid(True, alpha=0.3)

# multi-dimensonal analysis
ax7 = fig.add_subplot(gs[2, :])
pivot_data = df.pivot_table(values='Survived', index='Age', columns='Pclass', aggfunc='mean')
pivot_data_smooth = pivot_data.rolling(window=5, center=True).mean()

for i, pclass in enumerate([1, 2, 3]):
    if pclass in pivot_data_smooth.columns:
        ax7.plot(pivot_data_smooth.index, pivot_data_smooth[pclass], 
                color=colors[i], linewidth=2, label=f'Class {pclass}', alpha=0.8)
        
        
        rolling_std = pivot_data[pclass].rolling(window=10, center=True).std()
        upper = pivot_data_smooth[pclass] + rolling_std
        lower = pivot_data_smooth[pclass] - rolling_std
        ax7.fill_between(pivot_data_smooth.index, lower, upper, 
                        color=colors[i], alpha=0.2)

ax7.set_title('Survival Probability by Age and Class (with Confidence Intervals)', 
              fontweight='bold', fontsize=12)
ax7.set_xlabel('Age')
ax7.set_ylabel('Survival Probability')
ax7.legend()
ax7.grid(True, alpha=0.3)
ax7.set_ylim(0, 1)

# Professional styling
plt.suptitle('Comprehensive Titanic Dataset Analysis Dashboard', 
             fontsize=18, fontweight='bold', y=0.98)

fig.patch.set_facecolor('#FAFAFA')
for ax in fig.get_axes():
    ax.set_facecolor('#FFFFFF')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# statistical summary
print("\n" + "="*70)
print("TITANIC SURVIVAL ANALYSIS SUMMARY")
print("="*70)

overall_survival = df['Survived'].mean()
print(f"Overall Survival Rate: {overall_survival:.1%}")

print(f"\nSurvival by Gender:")
gender_survival = df.groupby('Sex')['Survived'].mean()
for gender, rate in gender_survival.items():
    print(f"  {gender.capitalize():8}: {rate:.1%}")

print(f"\nSurvival by Class:")
class_survival = df.groupby('Pclass')['Survived'].mean()
for pclass, rate in class_survival.items():
    print(f"  Class {pclass}: {rate:.1%}")

# stat test
male_survival = df[df['Sex'] == 'male']['Survived']
female_survival = df[df['Sex'] == 'female']['Survived']
chi2, p_value = stats.chi2_contingency(pd.crosstab(df['Sex'], df['Survived']))[:2]
print(f"\nGender-Survival Chi-square test p-value: {p_value:.2e}")

print("="*70)
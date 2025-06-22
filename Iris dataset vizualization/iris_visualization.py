import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# setting seaborn
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

#iris
df = sns.load_dataset('iris')

# species to categorical for better visualization
df['species'] = df['species'].astype('category')

def create_correlation_heatmap():
    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes(include=['float64']).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Mask for upper triangle
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                cbar_kws={'label': 'Correlation Coefficient'}, square=True)
    plt.title('Correlation Heatmap of Iris Features', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_pairplot():
    sns.pairplot(df, hue='species', palette='viridis', 
                 diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.suptitle('Pairplot of Iris Features by Species', y=1.02, fontsize=16)
    plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()

# violin plots for Feature Distributions
def create_violin_plots():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    titles = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    
    for i, (feature, title) in enumerate(zip(features, titles)):
        row, col = i // 2, i % 2
        sns.violinplot(x='species', y=feature, data=df, ax=axes[row, col], 
                       palette='muted', inner='quartile')
        axes[row, col].set_title(f'Distribution of {title} by Species')
        axes[row, col].set_xlabel('Species')
        axes[row, col].set_ylabel(title)
    
    plt.suptitle('Violin Plots of Iris Features by Species', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig('violin_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

#PCA Scatter Plot
def create_pca_scatter():
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.select_dtypes(include=['float64']))

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
    pca_df['Species'] = df['species']
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Species', 
                    palette='viridis', size='Species', sizes=(50, 200), 
                    alpha=0.7)
    plt.title('PCA Scatter Plot of Iris Data')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} Variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} Variance)')
    plt.legend(title='Species', loc='best')
    plt.tight_layout()
    plt.savefig('pca_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    create_correlation_heatmap()
    create_pairplot()
    create_violin_plots()
    create_pca_scatter()
    print("Visualizations saved as PNG files: correlation_heatmap.png, pairplot.png, violin_plots.png, pca_scatter.png")
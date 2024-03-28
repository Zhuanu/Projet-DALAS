import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import re

# -------------------------------------

file_path = 'data/immodataParis.csv'
df = pd.read_csv(file_path)

# Adresse,Ville,Arrondissement,Type,Prix (€),Prix mensuel (€),Pièce(s),Surface (m2),Date de vente

# -------------------------------------
# One-hot encoding

# Label encoding (for ordinal categorical variables)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Type'] = label_encoder.fit_transform(df['Type'])
df['Arrondissement'] = label_encoder.fit_transform(df['Arrondissement'])

# -------------------------------------
# Preprocessing

def extract_numerical(value):
    if isinstance(value, str):
        numerical_value = re.search(r'(\d+).*', value)
        if numerical_value:
            return int(numerical_value.groups(1)[0])
        else:
            return value
    else :
        return value
    
df['Surface (m2)'] = df['Surface (m2)'].apply(extract_numerical)

# -------------------------------------

# Assuming 'df' is your DataFrame, you might need to preprocess it if necessary
# For hierarchical clustering, it's common to standardize the data
new_df = df[["Arrondissement", "Type", "Prix (€)","Prix mensuel (€)","Pièce(s)","Surface (m2)"]] # .iloc[:800]
print([i for i in new_df['Surface (m2)']])
scaler = StandardScaler()
scaled_df = scaler.fit_transform(new_df)

# Perform hierarchical clustering
# Adjust parameters as needed, e.g., linkage method, number of clusters, etc.
clustering = AgglomerativeClustering(n_clusters=10, linkage='ward')
cluster_labels = clustering.fit_predict(scaled_df)

# Add cluster labels to DataFrame
new_df['Cluster'] = cluster_labels

# -------------------------------------

# Plot dendrogram (optional, requires scipy)
# from scipy.cluster.hierarchy import dendrogram, linkage
# linked = linkage(scaled_df, 'ward')
# dendrogram(linked)
# plt.show()

# -------------------------------------

# Visualize clusters (example with pairplot, adjust as needed)
sns.pairplot(new_df, hue='Cluster')
plt.show()
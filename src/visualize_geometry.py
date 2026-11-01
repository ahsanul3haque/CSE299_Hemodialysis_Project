import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def visualize_oversampling_geometry():
    print("Loading versioned datasets...")
    paths = {
        "Original": "data/versions/train_baseline_v1.csv",
        "SMOTE": "data/versions/train_smote_v1.csv",
        "ADASYN": "data/versions/train_adasyn_v1.csv"
    }

    datasets = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Check your data/versions/ folder.")
        datasets[name] = pd.read_csv(path)

    #Standardize and fit PCA ONLY on the original dataset to freeze the coordinate space
    print("Fitting PCA map on original data...")
    X_orig = datasets["Original"].drop(columns=["Hypotension"])
    
    scaler = StandardScaler()
    X_orig_scaled = scaler.fit_transform(X_orig)
    
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_orig_scaled)

    #Set up the visual canvas
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
    fig.suptitle("Geometric Analysis of Synthetic Oversampling (PCA Projection)", fontsize=18, fontweight='bold')
    
    palette = {0: "#0499FDC0", 1: "#E53935"} #Blue for Safe, Red for Crash

    #Process and plot each dataset
    for ax, (name, df) in zip(axes, datasets.items()):
        print(f"Projecting and plotting {name} data...")
        
        X = df.drop(columns=["Hypotension"])
        y = df["Hypotension"]
        
        #Transform data using the original scaler and PCA map
        X_pca = pca.transform(scaler.transform(X))
        
        sns.scatterplot(
            x=X_pca[:, 0], y=X_pca[:, 1], 
            hue=y, palette=palette, 
            alpha=0.3, edgecolor=None, s=15, ax=ax
        )
        
        #Formatting
        ax.set_title(f"{name} Distribution\n(Total Rows: {len(df)})", fontsize=14)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend(title="Hypotension", labels=["0: Safe", "1: Crash"])
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    
    os.makedirs("visualizations", exist_ok=True)
    save_path = "visualizations/oversampling_geometry_pca.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Visualization saved to: {save_path}")
if __name__ == "__main__":
    visualize_oversampling_geometry()
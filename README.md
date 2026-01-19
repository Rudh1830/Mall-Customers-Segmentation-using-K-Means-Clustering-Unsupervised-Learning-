# Mall-Customers-Segmentation-using-K-Means-Clustering-Unsupervised-Learning-
📌 Project Overview

Algorithm Used: K-Means Clustering

Learning Type: Unsupervised Learning

Dataset: Mall Customers Dataset

Tools & Libraries: Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

Objective: Group customers into clusters based on similarity without using labels

📂 Dataset Information

The dataset contains the following features:

Column Name	Description
CustomerID	Unique ID for each customer
Genre	Gender of the customer
Age	Age of the customer
Annual_Income_(k$)	Annual income in thousand dollars
Spending_Score_(1-100)	Spending behavior score
⚙️ Project Workflow

Data Loading

Loaded dataset using Pandas

Checked shape, info, null values, and basic statistics

Data Preprocessing

Encoded categorical variable (Genre) using LabelEncoder

Verified missing values

Prepared data for clustering

Exploratory Data Analysis (EDA)

Inspected distributions

Checked feature summaries

Verified data quality

Finding Optimal Number of Clusters

Used the Elbow Method

Calculated WCSS (Within-Cluster Sum of Squares) for 1–10 clusters

Selected k = 5 based on the elbow curve

Model Training

Trained KMeans with 5 clusters

Assigned cluster labels to each customer

Results

Successfully segmented customers into 5 distinct groups

Calculated final inertia value

Stored clustered data in a new DataFrame

📈 Elbow Method Visualization

The Elbow Method was used to identify the optimal number of clusters by plotting WCSS against the number of clusters.

plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
🧪 Model Implementation
kmeans = KMeans(n_clusters=5, random_state=42)
d['cluster'] = kmeans.fit_predict(d)

Each customer is now assigned a cluster label based on similarity.

🧠 Key Learnings

How unsupervised learning works without target labels

Importance of preprocessing before clustering

Using the Elbow Method to choose optimal k

Interpreting clustering results for business insights

⚠️ Notes & Limitations (Important)

All features were used directly without scaling
(K-Means is sensitive to feature scale; scaling would improve results)

CustomerID was included in clustering, which is not ideal

Clustering quality could be improved by selecting only relevant features

Visualization of clusters is not included (can be added using 2D plots or PCA)

🚀 Future Improvements

Apply MinMaxScaler or StandardScaler

Remove irrelevant features like CustomerID

Visualize clusters using PCA

Compare with other clustering algorithms (DBSCAN, Hierarchical)

Interpret clusters for marketing insights

🛠️ How to Run
pip install numpy pandas matplotlib seaborn scikit-learn
python mall_kmeans.py

---------------------------------------------------------------------------------
👨‍💻 Author

Rudresh

Student 

Machine Learning Enthusiast

📌 Project for learning Unsupervised ML & Clustering Techniques

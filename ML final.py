import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
import seaborn as sns
from math import pi


#Load Data

df=pd.read_csv("clean_student_performance_final.csv")
df["month"]=pd.to_datetime(df["month"])
df["month"]=df["month"].dt.month.astype(int)

cols_to_drop=["student_id","name","grade_level","description","country","teacher_name"]
df=df.drop(columns=cols_to_drop,errors='ignore')


#Encoding categorical variables

encoding_maps={"gender":{"female":0,"male":1},
               "difficulty_level":{"easy":0,"medium":1,"hard":2},
               "parent_education":{"none":0,"high school":1,"college":2,"postgrad":3},
               "health_condition":{"normal":0,"mild illness":1,"chronic":2},
               "performance_level":{"weak":0,"average":1,"good":2,"excellent":3}}
for col,mapping in encoding_maps.items():
    if col in df.columns:
        df[col]=df[col].map(mapping)


#Target encoding for categorical variables

target_col="performance_level"
target_encode_cols=["subject_name","city","admission_year","free_time_activity","school_transport"]
for col in target_encode_cols:
    if col in df.columns:
        mapping=df.groupby(col)[target_col].mean()
        df[col]=df[col].map(mapping)


#Feature engineering

if {"score","study_hours"}.issubset(df.columns):
    df["efficiency_ratio"]=df["score"]/np.maximum(df["study_hours"],1)
if {"score","attendance_rate"}.issubset(df.columns):
    df["attendance_impact"]=df["score"]*df["attendance_rate"]
if {"score","homework_completion_rate"}.issubset(df.columns):
    df["homework_impact"]=df["score"]*df["homework_completion_rate"]


#Feature Selection for Clustering
exclude_cols=["performance_level","cluster","cluster_name"]
numeric_cols=df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols=[c for c in numeric_cols if c not in exclude_cols]

#Calculate variances for each coloumn
variances=df[numeric_cols].var().sort_values(ascending=False)
print("\nTop 10 features by variance:")
print(variances.head(10))

#Delete coloumns with correlation > 0.9
corr_matrix=df[numeric_cols].corr().abs()
upper_tri=corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
to_drop=[column for column in upper_tri.columns if any(upper_tri[column]>0.9)]

if to_drop:
    print("\n Highly correlated features removed:",to_drop)

#Select top features by variance
selected_features=[col for col in variances.index if col not in to_drop][:15]  


clustering_cols=selected_features.copy()


#Scale the selected features
scaler=StandardScaler()
df_cluster=scaler.fit_transform(df[clustering_cols])


#PCA for dimensionality reduction

pca=PCA(n_components=None)
pca.fit(df_cluster)
cumulative_variance=np.cumsum(pca.explained_variance_ratio_)
n_components_90=np.argmax(cumulative_variance>=0.90)+1
print(f"\nPCA reduced dimensions to: {n_components_90}")

df_pca=PCA(n_components=n_components_90).fit_transform(df_cluster)


#Optimal K selection using Silhouette Score

scores=[]
for k in range(2,11):
    km=KMeans(n_clusters=k,random_state=42,n_init=15)
    labels=km.fit_predict(df_pca)
    score=silhouette_score(df_pca,labels)
    scores.append((k,score))

best_k=max(scores,key=lambda x:x[1])[0]
kmeans=KMeans(n_clusters=best_k,random_state=42,n_init="auto")
best_labels=kmeans.fit_predict(df_pca)
best_score=silhouette_score(df_pca,best_labels)
print(f"Final Silhouette Score: {best_score:.4f}")
df["cluster"]=best_labels


#Cluster profiling & naming

cluster_profiles=df.groupby("cluster")[["score","efficiency","attendance_rate","study_hours","previous_gpa"]].mean(numeric_only=True)
cluster_names={}
for cluster_id in cluster_profiles.index:
    score=cluster_profiles.loc[cluster_id,"score"]
    study_hours=cluster_profiles.loc[cluster_id,"study_hours"]
    if score>85:
        if study_hours>cluster_profiles["study_hours"].median():
            cluster_names[cluster_id]="High Performers - Hard Working"
        else:
            cluster_names[cluster_id]="High Performers - Efficient"
    else:
        if cluster_profiles.loc[cluster_id,"attendance_rate"]>df["attendance_rate"].mean():
            cluster_names[cluster_id]="Developing Students - Regular"
        else:
            cluster_names[cluster_id]="Needs Support - Irregular"

df["cluster_name"]=df["cluster"].map(cluster_names)
print("\nCluster distribution:")
print(df["cluster_name"].value_counts())


#Visualization (PCA 2D, Boxplots, Radar)

pca_2d=PCA(n_components=2)
df_pca_2d=pca_2d.fit_transform(df_cluster)

plt.figure(figsize=(14,10))
plt.subplot(2,2,1)
sns.scatterplot(x=df_pca_2d[:,0],y=df_pca_2d[:,1],hue=df["cluster_name"],palette="Set2",s=60,alpha=0.8,edgecolor="black",linewidth=0.5)
plt.xlabel(f"PCA Component 1 ({pca_2d.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PCA Component 2 ({pca_2d.explained_variance_ratio_[1]:.2%})")
plt.title(f"Student Clusters\n(Silhouette Score: {best_score:.4f})")
plt.legend(title="Cluster")

plt.subplot(2,2,2)
sns.boxplot(data=df,x="cluster_name",y="score",hue="cluster_name",legend=False,palette="Set2")
plt.title("Score Distribution by Cluster")
plt.xticks(rotation=15)

plt.subplot(2,2,3)
sns.boxplot(data=df,x="gender",y="previous_gpa",palette=["#FF9999","#66CCFF"])
plt.title("GPA by Gender")
plt.xlabel("Gender (0=Female, 1=Male)")
plt.ylabel("Previous GPA")

plt.subplot(2,2,4)
perf_labels={0:"Weak",1:"Average",2:"Good",3:"Excellent"}
performance_pivot=pd.crosstab(df["cluster_name"], df["performance_level"])
performance_pivot.rename(columns=perf_labels,inplace=True)
performance_pivot.plot(kind="bar",ax=plt.gca(),color=["#FF9999","#FFCC99","#99CCFF","#66CC99"])
plt.title("Performance Level Distribution")
plt.xticks(rotation=15)
plt.ylabel("Number of Students")

plt.tight_layout()
plt.show()


#Radar chart

cluster_0_data=df[df["cluster"]==0]
cluster_1_data=df[df["cluster"]==1]

categories=["Score","Efficiency","Attendance","Homework","Study Hours"]
categories=[*categories,categories[0]]

def normalize(val,max_val):
    return val/max_val

cluster_0_vals=[normalize(cluster_0_data["score"].mean(),100),
                normalize(cluster_0_data["efficiency"].mean(),2),
                normalize(cluster_0_data["attendance_rate"].mean(),100),
                normalize(cluster_0_data["homework_completion_rate"].mean(),100),
                normalize(cluster_0_data["study_hours"].mean(),8)]
cluster_0_vals=[*cluster_0_vals, cluster_0_vals[0]]

cluster_1_vals=[normalize(cluster_1_data["score"].mean(),100),
                normalize(cluster_1_data["efficiency"].mean(),2),
                normalize(cluster_1_data["attendance_rate"].mean(),100),
                normalize(cluster_1_data["homework_completion_rate"].mean(),100),
                normalize(cluster_1_data["study_hours"].mean(),8)]
cluster_1_vals=[*cluster_1_vals, cluster_1_vals[0]]

label_loc=np.linspace(0,2*pi,len(categories))
fig,ax=plt.subplots(figsize=(8,8),subplot_kw=dict(projection="polar"))
ax.plot(label_loc,cluster_0_vals,label="Developing Students",color="#FF6B6B")
ax.plot(label_loc,cluster_1_vals,label="High Performers",color="#4ECDC4")
ax.fill(label_loc,cluster_0_vals,alpha=0.1,color="#FF6B6B")
ax.fill(label_loc,cluster_1_vals,alpha=0.1,color="#4ECDC4")
ax.set_xticks(label_loc)
ax.set_xticklabels(categories)
ax.set_title("Cluster Comparison Radar Chart",size=14,pad=20)
ax.legend(bbox_to_anchor=(1.2,1.0))
plt.tight_layout()
plt.show()
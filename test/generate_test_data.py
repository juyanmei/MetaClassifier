import pandas as pd
import numpy as np

# Generate profile data with 300 samples
np.random.seed(42)
samples = [f"sample{i}" for i in range(1, 301)]
species_data = np.random.dirichlet(np.ones(300), size=300)
# 确保每行总和为1
species_data = species_data / species_data.sum(axis=1, keepdims=True)
species_data = species_data.round(2)
prof_df = pd.DataFrame(species_data, columns=[f"species{i}" for i in range(1, 301)], index=samples)
prof_df.to_csv("test/prof_test.csv")


# Initialize arrays for metadata
projects = ["ProjectA"]*50 + ["ProjectB"]*50 + ["ProjectC"]*50 + ["ProjectD"]*50 + ["ProjectE"]*50 + ["ProjectF"]*50

# Assign 50% healthy/50% disease samples for each project
groups = []
for i in range(6):  # 6 projects
    groups.extend(["Health"]*25 + ["Disease"]*25)  # 50 samples per project
disease_types = ["DiseaseA"]*100 + ["DiseaseB"]*100 + ["DiseaseC"]*100
two_major_diseases = ["SubMajorDiseaseA"]*150 + ["SubMajorDiseaseB"]*150
one_major_disease = ["OneMajorDisease"] * 300

meta_df = pd.DataFrame({
    "SampleID": samples,
    "Group": groups,
    "Project": projects,
    "Disease": disease_types,
    "TwoMajorDisease": two_major_diseases,
    "OneMajorDisease": one_major_disease
})
meta_df.to_csv("test/metadata_test.csv", index=False)
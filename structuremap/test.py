# Import structuremap functions
import structuremap.utils
structuremap.utils.set_logger()
from structuremap.processing import calculate_distance_features, download_alphafold_cif, download_alphafold_pae, format_alphafold_data, calculate_pPSE, get_smooth_score, annotate_proteins_with_idr_pattern, get_extended_flexible_pattern, get_proximity_pvals, perform_enrichment_analysis, perform_enrichment_analysis_per_protein, evaluate_ptm_colocalization, extract_motifs_in_proteome
from structuremap.plotting import plot_enrichment, plot_ptm_colocalization

import pandas as pd
import numpy as np
import os
import re
import plotly.express as px
import tqdm
import tempfile
from joblib import Parallel, delayed

output_dir = "/mnt/nimble/data/output/abarthmaron/abpp/structuremap"

# df = pd.read_excel("../data/Venom_1-5.cysteine_hit_calling_combined_cysteine_label_annotated_revised_NB.xlsx", sheet_name="StringentFilters")
df = pd.read_excel('/mnt/nimble/data/output/abarthmaron/abpp/Venom_1-5.cysteine_hit_calling.20240514.plus_detection_rate.read_only.xlsx', sheet_name='full_dataset')

# unique_uniprot_ids = set(list(df['UNIPROT']))
# unique_uniprot_ids = list(df.loc[~df['hits_single_relaxed'].isna()]['UNIPROT'].unique())
unique_uniprot_ids = list(df['UNIPROT'].unique())
print(f"There are {len(unique_uniprot_ids)} unique uniprot IDs.")

os.path.join(output_dir, 'uniprot_ids.txt')

uniprot_id_file = os.path.join(output_dir, 'unique_uniprot_ids.txt')
with open(uniprot_id_file, 'w') as f:
    for id in unique_uniprot_ids:
        f.write(f'{id} ')

## Try with different approach and str.split into a list
txt_file = open(uniprot_id_file, "r")
file_content = txt_file.read()
print(f"File Content: {file_content}")

human_fasta_list = file_content.split()
print(f"\nTotal Protein Count: {len(human_fasta_list)}")
print(f"Total Unqiue Count: {len(set(human_fasta_list))}")
cif_dir = os.path.join(output_dir, 'cif')
pae_dir = os.path.join(output_dir, 'pae')

# output_dir = tempfile.gettempdir()
# human_fasta_list = ['Q9NRZ5', 'O43353','P24941','Q92918','P45984','P28482','O96017',
#                  'P02730','Q8NB16','Q13546','P29320','P08559','P15121']
# # human_fasta_list = ['O43353']
# cif_dir = os.path.join(output_dir, 'tutorial_cif')
# pae_dir = os.path.join(output_dir, 'tutorial_pae')

parallelize_download = True
if parallelize_download:
    # Determine the number of CPU cores to use 
    n_jobs = 300  

    # Split the list of human fasta IDs into sublists
    human_fasta_array = np.array_split(np.array(human_fasta_list), n_jobs)
    human_fasta_list = [list(x) for x in human_fasta_array]

    # Download cif files in parallel
    _ = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
        delayed(download_alphafold_cif)(hf_list, cif_dir) for hf_list in human_fasta_list
    )

    # Download pae files in parallel
    _ = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
        delayed(download_alphafold_pae)(hf_list, pae_dir) for hf_list in human_fasta_list
    )

    # Flatten the list of human fasta IDs
    human_fasta_list = [l for sublist in human_fasta_list for l in sublist]

parallelize_download = False
if parallelize_download:
    # Determine the number of CPU cores to use 
    n_jobs = 70

    # Split the list of human fasta IDs into sublists
    human_fasta_array = np.array_split(np.array(human_fasta_list), n_jobs)
    human_fasta_list = [list(x) for x in human_fasta_array]

    # Format AF data in parallel
    AF_annotation_list = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
        delayed(format_alphafold_data)(cif_dir, hf_list) for hf_list in human_fasta_list
    )

    # Compile annotation list into a single dataframe
    alphafold_annotation = pd.concat(AF_annotation_list)

    # Flatten the list of human fasta IDs
    human_fasta_list = [l for sublist in human_fasta_list for l in sublist]
else:
    alphafold_annotation = format_alphafold_data(
        directory=cif_dir, 
        protein_ids=human_fasta_list)
    
alphafold_annotation.to_csv(os.path.join(output_dir, '2025-01-09_alphafold_annotation.csv'), index=False)

for p in [pae_dir, None]:
    for dist in [9, 12, 24]:
        full_sphere_exposure = calculate_pPSE(
            df=alphafold_annotation, 
            max_dist=24, 
            max_angle=180, 
            error_dir=pae_dir)

        alphafold_accessibility = alphafold_annotation.merge(
            full_sphere_exposure, how='left', on=['protein_id','AA','position'])


for p in [pae_dir, None]:
    for dist in [9, 12, 15]:
        part_sphere_exposure = calculate_pPSE(
            df=alphafold_annotation, 
            max_dist=dist, 
            max_angle=70, 
            error_dir=p)

        alphafold_accessibility = alphafold_accessibility.merge(
            part_sphere_exposure, how='left', on=['protein_id','AA','position'])

merge_cols = ['protein_id', 'position', 'AA', 'quality']
for p in [pae_dir, None]:
    df = calculate_distance_features(alphafold_annotation['protein_id'].unique().tolist(), cif_dir, error_dir=p)
    overlap_cols = df.drop(columns=merge_cols).columns.intersection(alphafold_accessibility.columns)
    df = df.drop(columns=overlap_cols)
    alphafold_accessibility = alphafold_accessibility.merge(df, how='left', on=['protein_id', 'position', 'AA', 'quality'])

alphafold_accessibility['high_acc_5'] = np.where(alphafold_accessibility.nAA_12_70_pae <= 5, 1, 0)
alphafold_accessibility['low_acc_5'] = np.where(alphafold_accessibility.nAA_12_70_pae > 5, 1, 0)

alphafold_accessibility_smooth = get_smooth_score(
    alphafold_accessibility, 
    np.array(['nAA_24_180_pae']), 
    [10])

alphafold_accessibility_smooth['IDR'] = np.where(
    alphafold_accessibility_smooth['nAA_24_180_pae_smooth10']<=34.27, 1, 0)

alphafold_accessibility_smooth_pattern = annotate_proteins_with_idr_pattern(
    alphafold_accessibility_smooth,
    min_structured_length = 80, 
    max_unstructured_length = 20)

alphafold_accessibility_smooth_pattern_ext = get_extended_flexible_pattern(
    alphafold_accessibility_smooth_pattern, 
    ['flexible_pattern'], [5])

# alphafold_accessibility.to_csv(os.path.join(output_dir, '2024-10-03_AlphaFoldPredicted_Accessibility.csv'))

alphafold_accessibility_smooth_pattern_ext.to_csv(os.path.join(output_dir, '2025-01-09_AF2_structural_data.csv'), index=False)

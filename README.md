# adaptation_tnseq
Scripts to generate plots in ["Quantifying the adaptive landscape of commensal gut shared.BACTERIA using high-resolution lineage tracking"](https://www.biorxiv.org/content/10.1101/2022.05.13.491573v1) (Wong and Good, bioRxiv, 2022).

Python distribution: Python v3.9.7

Required libraries: 
- matplotlib v3.5.0, 
- numpy v1.20.3, 
- pickle v0.7.5, 
- scipy v1.9.1,
- scikit-learn v1.1.1

From installation, fully producing all plots should take <~30 minutes. However, some intermediate data files are large, and will require ~1.5 GB of disk space, and (similar amounts of RAM)
To generate all plots, unzip "pickled_data" in data/folder/

1) Unzip data in data/ (after, should contain files like: "BWH2_read_arrays.txt")
2) From {project_root}, run "python scripts/make_pickled_data_files.py"
3) Almost all individual plots (except Figs 4, S13, S14) may now be generated by running "python scripts/fig*.py", and should take <1 minute to run
4) Some scripts nominally rely on output of previous plot scripts---in particular, "fig1_muller_and_transition.py". But these are small files that are supplied in data.
5) If some script fails, because some "*.pkl" file is not found, try running fig1_muller_and_transition.py
6) Making Figs 4. S13,S14 --- "fitness profiling" plots ---require large intermediate files: run "python generate_fitness_profile_data.py" to generate.
7) Then execute scripts (which both take >1 minute to run): fig4A_sfig13_fitness_profiling_gene_ko_clusters.py, fig4B_sfig14_fitness_profiling_novel_clusters.py

Drift estimation is time-intensive. 
In SI of paper, all permutations of mice are considered, 
    and (for each permutation) n=1000 bootstraps are done to estimate uncertainty for each data point for weighted regression.
Here, as a demo, we speed up these simulations/inferences by considering only ~30-100 (of 1260) permutations of mice, and n=100 bootstraps are performed. 
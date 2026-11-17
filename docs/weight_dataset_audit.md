# Weight datasets reviewed for IACES

The four files were reviewed for compatibility with the fitness assessment classifier and workout calorie predictor. None contains both the A–D fitness class and the physical-test measurements required by the former, or the workout calorie labels and session features required by the latter. They were not inserted into either model.

- `Data.csv`: 10,257 rows of dated weight and coded survey records, with 1,338 unique participants. After parsing date and numeric weight, 7,185 rows remain; 756 participants have at least two numeric observations, and 537 have observations spanning at least 90 days. Some values require cleaning (including a zero weight).
- `Demographics.csv`: 1,411 participant records including gender and height; 1,331 participants with numeric dated weights also appear in both demographics and screening files. Gender distribution is 1,158 female, 249 male, and 4 missing. The participant ID permits a join; this does not establish generalization to the app's users.
- `Screening.csv`: 1,422 records with coded screening fields; use `Variable+key.xlsx` or the study codebook to establish definitions, units, and timing before modeling. These fields lack the current fitness model's class label.
- `Gym_Progress_Dataset.csv`: 200 daily rows without participant IDs. Among 199 consecutive-day intervals, 123 show weight changes greater than 2 kg; the median absolute daily change is 3.0 kg and the largest is 12.6 kg. This is a quality concern, and the file was excluded from training.

The research files may support a future, separate longitudinal outcome study with participant-level splits, validated measurement units, data quality rules, missingness and censoring analysis, and a held-out baseline. They do not establish a daily exercise recommendation or a reliable weight-goal completion date. Dataset origin, consent, privacy, and license should be verified before reuse.

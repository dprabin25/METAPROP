
METAPROP
MetaPropA is a browser-based Streamlit app for meta-analysis of single proportions (prevalence, incidence, event rates). Upload a CSV with one row per study, map three columns, and it produces pooled estimates, heterogeneity statistics, forest plots, funnel plots, publication-bias tests, trim-and-fill correction, meta-regression and correlation analysis. Every result can be exported as one structured ZIP.

Random-effects (DerSimonian–Laird τ²) · Hartung–Knapp–Sidik–Jonkman (HKSJ) correction · Logit transformation · Okabe–Ito colour-blind-safe palette

Contents
Features
Installation
Running the app
Input data format
Workflow
Statistical methods and why they matter
Exported ZIP structure
Interpreting results: quick reference
Limitations
References
Citation and license
Features
Tab	What it shows
Cleaned Data	Rows that passed validation, ready for analysis (downloadable CSV)
Excluded Data	Rows that were dropped, each with the reason (downloadable CSV)
Summary Stats	Random-effects and fixed-effect pooled proportions, 95% CI, 95% prediction interval, I², τ², Cochran's Q; per-study results and weights; per-group summaries
Forest (Overall)	Forest plot of every study with the pooled estimate
Forest (Subgroup)	Forest plot split by each selected grouping column
Funnel Plot	Funnel plots overall and per subgroup, with points optionally coloured by group
Publication Bias	Egger's regression test, Begg–Mazumdar rank correlation, and Duval–Tweedie trim-and-fill (L0 / R0) with an adjusted funnel plot
Meta-Regression	Weighted regression of the logit proportion on a numeric or categorical moderator
Correlations	Pearson or Spearman correlation heatmap across numeric study-level columns
You can also:

group by several columns at once; each gets its own subgroup analysis and its own folder in the export
exclude studies by hand or restrict the analysis to one subgroup value
set the number of decimal places for the whole app and for each tab
download each plot as a PNG, or generate one ZIP containing everything
Installation
You need Python 3.9 or newer.

git clone https://github.com/dprabin25/METAPROP.git
cd METAPROP
pip install -r requirements.txt
Dependencies: streamlit, pandas, numpy, scipy, matplotlib.

Running the app
streamlit run meta_analysis_app.py
Streamlit opens the app in your browser (by default at http://localhost:8501). If you don't upload a file, the app loads a built-in demo dataset, which you can also download from the sidebar to see the expected format.

Input data format
A CSV file with one row per study. Three columns are required, and you map them by hand in the sidebar (nothing is auto-detected):

Field	Meaning	Example
Study ID	Unique label for the study	Smith 2010
Sample (n)	Total number of participants	450
Cases / Events	Number with the outcome	85
Any other columns are optional. They can be used as grouping variables (e.g. Region, Study_Design), meta-regression moderators (e.g. Year, Mean_Age), or correlation variables.

Study_ID,Sample,Cases,Risk_Category,Year,Region,Mean_Age,Study_Design
Smith 2010,450,85,High risk,2010,North America,52.3,Cohort
Jones 2011,320,62,Low risk,2011,Europe,45.1,Case-Control
Chen 2012,680,110,High risk,2012,Asia,58.7,Cohort
Automatic data cleaning
Each row is checked, and a row is moved to Excluded Data with its reason if any check fails:

Check	Exclusion reason
Sample is blank	Sample missing
Sample is not a number (e.g. ABC)	Sample non-numeric
Cases is blank	Cases missing
Cases is not a number (e.g. NR)	Cases non-numeric
Sample ≤ 0	Sample <= 0
Cases < 0	Cases < 0
Cases > Sample	Cases > Sample
Why this matters: in a proportion meta-analysis a single impossible row (more cases than participants, or text in a numeric field) can break the analysis or quietly distort the pooled estimate. The app shows exactly what was removed and why, which makes the analysis reproducible and gives you what you need for a PRISMA flow diagram. At least 2 valid studies are needed to run the analysis.

Workflow
Upload your CSV in the sidebar, or use the demo data.
Map the Study ID, Sample (n) and Cases / Events columns.
(Optional) Choose one or more Group by columns.
(Optional) Exclude individual studies, or focus on one value of the first grouping column.
Look through the nine tabs.
Open ⚙️ Output controls to set display precision and Generate ZIP for a full export.
Statistical methods and why they matter
All calculations are done on the logit scale and back-transformed to proportions for display.

1. Logit transformation
For each study i with eᵢ events out of nᵢ participants:

y
i
=
logit
(
p
i
)
=
ln
⁡
 ⁣
(
e
i
n
i
−
e
i
)
,
v
i
=
1
e
i
+
1
n
i
−
e
i
y 
i
​
 =logit(p 
i
​
 )=ln( 
n 
i
​
 −e 
i
​
 
e 
i
​
 
​
 ),v 
i
​
 = 
e 
i
​
 
1
​
 + 
n 
i
​
 −e 
i
​
 
1
​
 
The pooled estimate and its interval are back-transformed with the inverse logit, 
p
=
1
/
(
1
+
e
−
y
)
p=1/(1+e 
−y
 ).

Why it matters: raw proportions are bounded between 0 and 1, and their variance depends on the proportion itself: it is squeezed near 0% and 100%. Pooling raw proportions can therefore give confidence intervals that run below 0% or above 100%, and can give too much weight to studies with extreme proportions. On the logit scale the values are unbounded and closer to normally distributed, so the pooled result and its CI always stay inside a valid range after back-transformation.

2. Continuity correction for zero and full events
If a study has 0 events or all events (eᵢ = 0 or eᵢ = nᵢ), 0.5 is added to the events and 1 to the sample (e + 0.5, n + 1) before the logit is taken.

Why it matters: the logit of 0 or 1 is infinite and the variance formula divides by zero. Without a correction these studies would crash the analysis or have to be thrown away, and throwing away studies with rare (or universal) outcomes biases the pooled estimate. The 0.5 correction keeps them in with a finite, conservative variance.

3. Fixed-effect (inverse-variance) model
w
i
=
1
v
i
,
μ
^
F
E
=
∑
w
i
y
i
∑
w
i
w 
i
​
 = 
v 
i
​
 
1
​
 , 
μ
^
​
  
FE
​
 = 
∑w 
i
​
 
∑w 
i
​
 y 
i
​
 
​
 
Why it matters: the fixed-effect model assumes every study estimates the same true proportion, so larger, more precise studies get more weight. It is shown as a reference point next to the random-effects result. When it differs a lot from the random-effects estimate, that is itself a sign of heterogeneity or of small-study effects.

4. Heterogeneity: Cochran's Q, I² and τ²
Q
=
∑
w
i
(
y
i
−
μ
^
F
E
)
2
∼
χ
k
−
1
2
Q=∑w 
i
​
 (y 
i
​
 − 
μ
^
​
  
FE
​
 ) 
2
 ∼χ 
k−1
2
​
 
I
2
=
max
⁡
 ⁣
(
0
,
 
Q
−
(
k
−
1
)
Q
)
×
100
%
I 
2
 =max(0,  
Q
Q−(k−1)
​
 )×100%
τ
D
L
2
=
max
⁡
 ⁣
(
0
,
 
Q
−
(
k
−
1
)
∑
w
i
−
∑
w
i
2
/
∑
w
i
)
τ 
DL
2
​
 =max(0,  
∑w 
i
​
 −∑w 
i
2
​
 /∑w 
i
​
 
Q−(k−1)
​
 )
Statistic	What it tells you
Cochran's Q (and its p-value)	Whether the studies differ more than chance alone would explain. It has low power when there are few studies and becomes almost always significant when there are many, so don't rely on the p-value alone.
I²	The percentage of total variation that comes from real differences between studies rather than sampling error. A rough guide: 25% is low, 50% moderate, 75% high.
τ² (DerSimonian–Laird)	The absolute between-study variance on the logit scale. It feeds the random-effects weights and the prediction interval.
Why it matters: in proportion meta-analyses, heterogeneity is the rule rather than the exception. Prevalence differs across populations, settings, diagnostic criteria and time periods, and I² above 90% is common. A pooled number reported without its heterogeneity can be badly misleading. These statistics tell you how far the single summary figure can be trusted and whether you should look for the sources with subgroup analysis or meta-regression.

5. Random-effects model (DerSimonian–Laird)
W
i
=
1
v
i
+
τ
^
2
,
μ
^
R
E
=
∑
W
i
y
i
∑
W
i
W 
i
​
 = 
v 
i
​
 + 
τ
^
  
2
 
1
​
 , 
μ
^
​
  
RE
​
 = 
∑W 
i
​
 
∑W 
i
​
 y 
i
​
 
​
 
Why it matters: the random-effects model assumes each study has its own true proportion, drawn from a distribution of true proportions. Because τ² is added to every study's variance, the weights become more even, so one very large study cannot dominate the result. This is the primary model in MetaPropA and the usual choice for prevalence studies, where the assumption of one shared true value is rarely credible.

6. Hartung–Knapp–Sidik–Jonkman (HKSJ) adjustment
The confidence interval of the random-effects estimate uses a rescaled variance and a t-distribution in place of the normal:

q
=
max
⁡
 ⁣
(
1
,
 
∑
W
i
(
y
i
−
μ
^
R
E
)
2
k
−
1
)
,
Var
(
μ
^
)
=
q
∑
W
i
q=max(1,  
k−1
∑W 
i
​
 (y 
i
​
 − 
μ
^
​
  
RE
​
 ) 
2
 
​
 ),Var( 
μ
^
​
 )= 
∑W 
i
​
 
q
​
 
95
%
 CI
=
μ
^
±
t
0.975
,
 
k
−
1
Var
(
μ
^
)
95% CI= 
μ
^
​
 ±t 
0.975,k−1
​
  
Var( 
μ
^
​
 )
​
 
(The max(1, ·) floor means the HKSJ interval is never narrower than the standard DerSimonian–Laird interval.)

Why it matters: the standard DerSimonian–Laird interval treats τ² as if it were known exactly. With a small or moderate number of studies this makes the intervals too narrow, so they cover the true value less often than 95% of the time. Simulation studies show that HKSJ gives coverage much closer to the stated 95%, and it is now recommended by methodological guidance such as the Cochrane Handbook. The result is a more honest statement of uncertainty.

7. 95% prediction interval
95
%
 PI
=
μ
^
±
t
0.975
,
 
k
−
2
τ
^
2
+
Var
(
μ
^
)
95% PI= 
μ
^
​
 ±t 
0.975,k−2
​
  
τ
^
  
2
 +Var( 
μ
^
​
 )
​
 
Why it matters: the confidence interval describes the uncertainty in the average proportion. The prediction interval describes where the true proportion of a new study or setting is likely to fall. When heterogeneity is high, the PI can be far wider than the CI, and for anyone applying the result to their own population that is often the more useful number. Reporting it is widely recommended alongside random-effects results.

8. Per-study estimates and weights
Each study's proportion is shown with its own 95% CI, 
expit
(
y
i
±
1.96
v
i
)
expit(y 
i
​
 ±1.96 
v 
i
​
 
​
 ), and its percentage weight in the random-effects model.

Why it matters: you can see which studies drive the pooled result, spot outliers, and check that no single study carries an unreasonable share of the weight.

9. Subgroup analysis
For every selected Group by column, the full random-effects analysis (pooled proportion, CI, I², τ², Q) is run separately within each group and shown in a subgroup forest plot, a group summary table and per-group funnel plots.

Why it matters: subgroup analysis is the most direct way to explain heterogeneity. If pooled prevalence in high-risk groups differs clearly from low-risk groups, and heterogeneity inside each group is lower, the grouping variable accounts for part of the variation. Subgroup results are observational comparisons between studies, so treat them as hypothesis-generating, and be careful with groups that contain very few studies.

10. Forest plots
Each study is drawn as a square (sized by weight) with its 95% CI. The pooled estimate is drawn as a diamond, with the heterogeneity statistics alongside.

Why it matters: the forest plot is the standard way to present a meta-analysis. It shows each study's estimate, its precision, how much the studies overlap, and the summary result, all on one figure.

11. Funnel plots
Each study's logit proportion is plotted against its standard error, with pseudo-95% confidence limits (
μ
^
±
1.96
 
S
E
μ
^
​
 ±1.96SE).

Why it matters: without bias, studies should spread symmetrically around the pooled estimate, with small (high-SE) studies scattered more widely at the bottom of the funnel. An asymmetric funnel, for example missing small studies on one side, suggests publication bias or other small-study effects. The plot is a visual check that the formal tests below put numbers on.

12. Egger's regression test
The standardised effect 
y
i
/
S
E
i
y 
i
​
 /SE 
i
​
  is regressed on precision $1/SE_i$:

y
i
S
E
i
=
β
0
+
β
1
⋅
1
S
E
i
+
ε
i
SE 
i
​
 
y 
i
​
 
​
 =β 
0
​
 +β 
1
​
 ⋅ 
SE 
i
​
 
1
​
 +ε 
i
​
 
The test asks whether the intercept β₀ differs from zero (t-test with k − 2 df).

Why it matters: a non-zero intercept means small studies report systematically different proportions from large ones, which is the statistical signature of funnel asymmetry. Egger's test is the most widely used formal test for small-study effects.

13. Begg–Mazumdar rank correlation test
Kendall's τ between the study effects 
y
i
y 
i
​
  and their standard errors 
v
i
v 
i
​
 
​
 .

Why it matters: this is a non-parametric counterpart to Egger's test. It needs fewer distributional assumptions but has lower power. Running both gives a more robust picture, and the app tells you when the two disagree.

Significance threshold. Both bias tests use p < 0.10 rather than 0.05, the convention for these tests because they have low statistical power. Both need at least 3 studies. They become much more reliable with 10 or more studies, and with fewer, results should be read cautiously. For proportions, funnel asymmetry can also come from genuine heterogeneity, so a significant result is a flag to look into, not proof of publication bias.

14. Trim-and-fill (Duval & Tweedie, 2000)
When Egger's or Begg's test is significant, trim-and-fill runs automatically; otherwise you can run it on request. The procedure:

Detects which side of the funnel has the excess studies, using an Egger-type regression.
Repeatedly trims the most extreme studies on that side and re-estimates the centre until the number of "missing" studies, k₀, stops changing.
Fills in k₀ mirror-image studies on the opposite side.
Re-pools all studies to give an adjusted proportion and CI, shown next to the original and on an adjusted funnel plot.
Two estimators are available:

L0 (default, recommended): based on the signed-rank sum; stable in most situations.
R0: based on the run of the largest positive ranks; can be unstable. If it fails to converge, or tries to impute more than 60% of the studies, the app applies no correction and suggests L0.
The formulas follow the reference implementation in the R metafor package.

Why it matters: the bias tests only tell you that asymmetry may exist. Trim-and-fill estimates how much it could change your conclusion. If the adjusted estimate is close to the original, the finding is robust to plausible publication bias. If it moves substantially, report that clearly. Trim-and-fill is a sensitivity analysis, not a "true" corrected answer.

15. Meta-regression
A weighted least-squares regression of the logit proportion on a study-level moderator, using random-effects weights 
W
i
=
1
/
(
v
i
+
τ
^
2
)
W 
i
​
 =1/(v 
i
​
 + 
τ
^
  
2
 ):

Numeric moderators (e.g. year, mean age): slope and intercept, with a bubble plot (bubble size = weight) and the fitted curve on the proportion scale.
Categorical moderators (e.g. region, design): dummy-coded against the first category, with a bar chart of group means.
Reported statistics:

Output	What it tells you
Coefficients, SE, t, p	Whether, and in which direction, the moderator is associated with the proportion (on the logit scale)
Q_res (and p-value)	Heterogeneity left over after accounting for the moderator
τ²_res	Residual between-study variance
R²	
(
1
−
τ
r
e
s
2
/
τ
2
)
×
100
%
(1−τ 
res
2
​
 /τ 
2
 )×100%: the share of between-study variance explained by the moderator
Why it matters: meta-regression extends subgroup analysis to continuous variables and quantifies how much heterogeneity a factor explains. For example: does prevalence rise with publication year? Do older populations show higher rates? As a rule of thumb you need around 10 studies per moderator, and the associations are ecological (between studies, not between individuals).

16. Correlation analysis
A Pearson (linear) or Spearman (rank-based, robust to outliers and non-linearity) correlation matrix across the numeric study-level variables you choose, shown as an annotated heatmap.

Why it matters: before meta-regression, it helps to know which moderators move together (e.g. year and sample size, or age and follow-up time). Strongly correlated moderators are confounded: their effects can't be separated. Spotting this early stops you over-interpreting a single-moderator regression.

Exported ZIP structure
⚙️ Output controls → Generate ZIP builds one archive named after your mapped columns:

{Sample}__{Cases}/
├── data/
│   ├── cleaned_data.csv
│   └── excluded_data.csv
├── overall/
│   ├── summary_stats.csv          # RE (DL + HKSJ) and FE pooled results, I², τ², Q
│   ├── per_study_results.csv      # proportion, 95% CI, weight per study
│   ├── publication_bias.csv       # Egger, Begg, trim-and-fill (L0)
│   ├── forest_overall.png
│   ├── funnel_overall.png
│   └── funnel_trimfill.png
└── subgroups/
    └── {group_column}/
        ├── group_summary.csv
        ├── forest_subgroup.png
        ├── funnel_subgroups.png
        └── {group_value}/
            ├── forest.png
            └── stats.csv
The ZIP downloads to your browser's default downloads folder.

Interpreting results: quick reference
Result	Meaning
Pooled proportion (RE)	Best estimate of the average proportion across studies
95% CI	Uncertainty around that average (HKSJ-adjusted)
95% PI	Expected range of the true proportion in a new study or setting
I² < 25% / 50% / > 75%	Low / moderate / high heterogeneity
Q p-value < 0.10	Evidence of heterogeneity beyond chance
Egger or Begg p < 0.10	Possible funnel asymmetry (small-study effects or publication bias)
Trim-and-fill k₀ = 0	No asymmetry to correct; estimate unchanged
Adjusted ≈ original	Conclusion robust to plausible publication bias
Meta-regression p < 0.05, high R²	Moderator explains a meaningful part of the heterogeneity
Limitations
Uses the logit transformation only. Freeman–Tukey double-arcsine and GLMM (random-intercept logistic) models are not implemented.
τ² is estimated with DerSimonian–Laird only (no REML or Paule–Mandel).
Publication-bias tests and meta-regression have low power with fewer than about 10 studies.
Meta-regression handles one moderator at a time.
Subgroup and meta-regression findings are between-study (ecological) associations, not causal effects.
References
DerSimonian R, Laird N. Meta-analysis in clinical trials. Controlled Clinical Trials. 1986;7(3):177–188.
Hartung J, Knapp G. A refined method for the meta-analysis of controlled clinical trials with binary outcome. Statistics in Medicine. 2001;20(24):3875–3889.
Sidik K, Jonkman JN. A simple confidence interval for meta-analysis. Statistics in Medicine. 2002;21(21):3153–3159.
Higgins JPT, Thompson SG. Quantifying heterogeneity in a meta-analysis. Statistics in Medicine. 2002;21(11):1539–1558.
Higgins JPT, Thompson SG, Spiegelhalter DJ. A re-evaluation of random-effects meta-analysis. JRSS A. 2009;172(1):137–159.
Egger M, Davey Smith G, Schneider M, Minder C. Bias in meta-analysis detected by a simple, graphical test. BMJ. 1997;315:629–634.
Begg CB, Mazumdar M. Operating characteristics of a rank correlation test for publication bias. Biometrics. 1994;50(4):1088–1101.
Duval S, Tweedie R. Trim and fill: a simple funnel-plot–based method of testing and adjusting for publication bias in meta-analysis. Biometrics. 2000;56(2):455–463.
Thompson SG, Higgins JPT. How should meta-regression analyses be undertaken and interpreted? Statistics in Medicine. 2002;21(11):1559–1573.
Viechtbauer W. Conducting meta-analyses in R with the metafor package. Journal of Statistical Software. 2010;36(3):1–48.
Higgins JPT, et al. (eds). Cochrane Handbook for Systematic Reviews of Interventions, version 6. Cochrane; 2023.
Citation and license
© 2026 Prabin Dawadi (under review). All rights reserved.

If you use MetaPropA in your research, please cite this repository:

Dawadi P. METAPROP / MetaPropA: a Streamlit application for proportion meta-analysis.
GitHub: https://github.com/dprabin25/METAPROP

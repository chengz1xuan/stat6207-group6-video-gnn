"""Generate IEEEtran A4 conference source directly from final measured artifacts."""
from pathlib import Path
import json,shutil,re
R=Path(__file__).resolve().parents[1];D=R/'artifacts/v2';load=lambda f:json.loads((D/f).read_text());m=load('metrics.json');a=load('audit.json');u=load('uncertainty.json');ga=load('graph_audit.json');lo=load('leave_brand_out_summary.json');o=load('optimization.json');cv=load('cross_validation.json');sp=load('split_policy.json');gnn=m['selected_primary_gnn'];v=m['test'][gnn]
def tex(s):
 return ''.join({'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','\\':r'\textbackslash{}'}.get(c,c) for c in str(s))
def f(x):return '--' if x is None else f'{x:.3f}'
labels={'Mean':'Mean (constant)','Metadata Ridge':'Metadata Ridge','Pooled Ridge':'Pooled-feature Ridge','Pooled MLP':'No-message MLP','Subgraph structural':'Subgraph: structural','Subgraph + similarity':'Subgraph: structural+sim.','Node structural':'Video node: structural','Node + similarity':'Video node: structural+sim.'}
rows=[]
for n,r in m['test'].items():
 line=' & '.join([labels[n],f(r['log_MAE']),f(r['log_RMSE']),f(r['Spearman'])])+r' \\'
 if n==gnn:line=r'\bfseries '+line
 rows.append(line)
lrows=[]
for n,r in lo['models'].items():lrows.append(' & '.join([labels[n],f(r['macro_log_MAE']),f(r['macro_log_RMSE']),f(r['macro_Spearman'])])+r' \\')
q=m['OOF_90_absolute_log_residual'][gnn];case=next(c for c in o['dataset_cases'] if c['changes']);new=o['new_video'];ci=u['test_log_RMSE_95'][gnn];diff=u['paired_log_MAE_similarity_minus_structural_95'];epochs='; '.join(labels[n]+': '+str(c['epochs']) for n,c in m['configs'].items() if 'epochs' in c)
source=r'''\documentclass[a4paper,conference]{IEEEtran}
\usepackage{amsmath,amssymb,graphicx,booktabs,url}
\usepackage[hidelinks]{hyperref}
\title{Advertising Video Popularity Prediction:\\Creative Subgraphs, Structural Edges, and Transfer}
\author{\IEEEauthorblockN{Group 6}\IEEEauthorblockA{STAT6207: Applied Deep Learning\\The Chinese University of Hong Kong}}
\begin{document}
\maketitle
\begin{abstract}
We test whether advertising creative relationships help predict YouTube popularity using a compact graph neural network (GNN). From the latest supplied brand packages, a frozen public-metadata collection recovers exact view counts for @@N@@ of 219 videos across 14 brands. The primary design represents each video as a creative/evidence subgraph, beginning with supplied structural edges; semantic edges are an ablated extension. A campaign-grouped 80/20 split, training-only grouped tuning, common metrics, no-message controls, and leave-one-brand-out tests separate predictive performance from demonstration. The selected subgraph GNN achieves log-view MAE @@MAE@@, RMSE @@RMSE@@, and Spearman @@RHO@@; the video-node structural extension reaches RMSE @@NODE@@. Added similarity does not establish a reliable gain, and cross-brand errors remain large. An evidence-linked web tool presents six held-out cases and a new-video storyboard. Recommendations are observational hypotheses, not demonstrated causal gains.
\end{abstract}
\begin{IEEEkeywords}
Graph neural networks, advertising, video popularity, grouped evaluation, creative recommendation.
\end{IEEEkeywords}

\section{Objective and Data}
The objective is to predict directly observable audience response and examine whether creative relationships add useful signal. Views measure cumulative reach, rather than sales, brand recognition, or intrinsic creative quality. We start with total views and target $y_i=\ln(1+v_i)$, which reduces the influence of the long popularity tail. The slide-level compact GNN is implemented as the baseline; no executable VideoGNN code, weights, or identifiable 120-video reference subset are supplied. Thus we use the documented latest complete inventory rather than inventing a reproduction subset.

\subsection{Inventory, Labels, and Fresh Collection}
Uniformly retaining the latest version per brand yields 219 unique URLs and 14 brands: HSBC and Hang Seng use v0.2; MTR actually contains 14 records. The original label audit finds 219 views and 184 likes, with no non-null comments. Most labels reside in each package's \texttt{evidence\_nodes.csv}: filter \texttt{kind=engagement\_stats}, parse JSON from \texttt{value\_json} or \texttt{text}, and read \texttt{views/likes/comments}. McDonald's also has direct columns in its video table. The original plan's 20-label estimate was therefore incomplete.

To meet the platform-collection requirement, we make one unauthenticated pass over all 219 supplied YouTube watch URLs on 9 November 2026, 12:01:03--12:02:18 UTC. Exact counts are extracted only from the current video's public primary metadata, never recommendation cards. We recover 214 view counts and channel identities, and 175 exact likes. No exact comment counts surface in the initial public metadata; these remain missing, not zero. Subscriber strings are rounded descriptive metadata, excluded from prediction. Public metadata may be available even when playback requires sign-in; no login barrier is bypassed. Five URLs without fresh exact views are excluded from the revised supervised cohort; archived counts are never substituted. There is no added external holdout.

Each retrieval preserves URL/ID, UTC time, HTTP/extraction status, SHA-256 response digest, exact evidence snippets, and field paths in per-video JSON. The frozen CSV, failure list and data dictionary are supplied. Publication dates and durations come from the original inventory; newly verified upload dates were not exposed. Exposure age uses the fresh capture time and supplied publication date, so date accuracy remains a limitation. No response label is approximated from \texttt{fit\_score} or creative annotations.

\section{Video Subgraphs and Model Baselines}
\subsection{Primary Graph: Supplied Structure First}
Each video is one graph, with a video/context root, creative-concept nodes, and selected textual-evidence nodes. We normalize schema aliases, allowlist creative fields, deduplicate text, and cap repeated visual/audience aspects at four and each selected evidence kind at three. We exclude engagement statistics, assessment/risk scores, quality flags, and sentences containing response-related terms. The resulting @@N@@ subgraphs contain @@GNODES@@ nodes and @@GEDGES@@ original directed relations before message-passing symmetrization.

Core edges are copied from \texttt{candidate\_edges.csv}: \texttt{video\_has\_aspect}, \texttt{video\_has\_evidence}, and \texttt{aspect\_supported\_by\_evidence}. Original edge IDs and packages are retained. Membership is deterministic; evidence-support links may be source-derived heuristics and are flagged. Cross-video heuristic suggestions are not silently treated as validated internal relations. The model symmetrizes retained edges for aggregation; it does not use view counts to construct them.

\subsection{Features and Similarity Ablation}
The root encodes brand and seven creative categories, log-duration/exposure age with missingness flags, and a 16-dimensional TF-IDF/SVD representation of message, hook, promise and CTA. Each node also has node-type/field indicators and a separate 16-dimensional TF-IDF/SVD text representation. Vocabulary, category encoding, medians, scaling, and SVD are fitted exclusively on the relevant training rows/nodes. Missing/unseen values have explicit handling. Simple word tokenization limits Chinese semantic fidelity.

The extension adds within-video semantic edges between nonroot nodes: cosine similarity in the training-fitted node-text SVD space must be at least 0.55, retaining up to two neighbors per node, then symmetrizing and deduplicating. This rule, node caps, and feature allowlists are fixed before evaluation. These edges are explicitly heuristic. The matched structural-only model differs only by their absence.

\subsection{Compact GNN and Non-Graph Controls}
Using standard PyTorch Geometric \texttt{SAGEConv} mean aggregation \cite{sage,pyg},
\begin{equation}
 h_u^{(\ell+1)}=\operatorname{ReLU}\!\left(W_s^{(\ell)}h_u^{(\ell)}+W_n^{(\ell)}\operatorname{mean}_{w\in N(u)}h_w^{(\ell)}+b^{(\ell)}\right).
\end{equation}
Two 24-unit layers yield a representation concatenating the root embedding with the graph's mean node embedding. A linear head predicts standardized log views. Training minimizes mean squared error; Adam uses learning rate 0.01. No views, likes, comments, or \texttt{fit\_score} enter node features.

Controls are a training-log-mean predictor; metadata Ridge (brand, duration, age); Ridge on concatenated raw root/mean node features; and a no-message MLP with the same node inputs, two 24-unit transformations, and identical root/mean readout. The GNN has extra neighbor weights, so the control is information-matched, not exactly parameter-matched.

\subsection{Video-Node Extension}
A second representation uses one video per node and the same global root features. Core links are original same-brand/author/creative-category video relations from supplied candidate edges. The extension adds five cosine-nearest training-feature links. Held-out nodes receive messages only from training references; there are no held-out-to-training or held-out-to-held-out messages. Two mean GraphSAGE layers predict log views. This extension tests another graph definition without concealing a stronger or weaker result behind model complexity.

\section{Quantitative Training and Evaluation}
\subsection{Shared Campaign-Grouped 80/20 Protocol}
Groups union normalized same-brand campaign labels and exact same-brand message duplicates. Of 214 videos, @@NGROUPS@@ groups remain. A fixed grouped split (seed @@SEED@@) places @@TRAIN@@ videos (79.91\%) in training and @@TEST@@ (20.09\%) in testing: the rounded 80/20 count. Every main model uses exactly these videos and metrics. Group and URL disjointness are asserted. This tests unseen campaign groups within known brands; paraphrased variants may escape the heuristic grouping, so it is not a guarantee of entirely unseen creative families.

Three grouped folds inside the outer training partition select Ridge $\alpha\in\{10,100\}$ and neural weight decay $\lambda\in\{0.01,0.1\}$. Each fold refits preprocessing and graph rules, trains for at most 260 epochs, and uses fold validation MSE for early stopping (patience 35). Hyperparameters minimize mean fold log RMSE. The final model refits all 171 training videos for the median selected-fold stopping epoch. Thus the held-out 20\% is not used for fitting, stopping, model selection, or edge tuning. Selected subgraph+similarity training uses @@EPOCH@@ epochs and $\lambda=@@WD@@$. Full fold scores, final training loss trajectories, configurations and source hashes are exported. CV early-stopping scores are tuning criteria, not unbiased generalization estimates.

Report MAE/RMSE on natural-log views and Spearman correlation. Raw-view errors use $\exp(\hat y)-1$, clipped to nonnegative counts; this back-transform is not an unbiased arithmetic expectation. Quantitative losses, fold errors and epochs are recorded rather than calling a trained network ``successful'' without a target. All baseline targets and metrics match the GNN.

\begin{table}[t]
\caption{Same-split held-out performance ($n=43$); lower errors are better. Bold identifies the training-CV-selected primary GNN, not the best test model.}
\label{tab:main}\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrr}\toprule
Model & Log MAE & Log RMSE & $\rho$\\\midrule
@@ROWS@@
\bottomrule\end{tabular}
\end{table}

\subsection{Uncertainty and Cross-Brand Transfer}
We draw 2,000 bootstrap resamples of test campaign groups (seed 44), retaining the same resample across models for paired differences. Intervals condition on the fixed split, collection and fitted model; they exclude split/model-selection uncertainty. Three additional GNN initializations probe training variability without replacing the primary seed 42.

For leave-one-brand-out (LOBO), each of 14 brands is held out in turn. All preprocessing and models are refitted on the other brands. A grouped inner holdout chooses epochs; the remaining brands are then refitted. To bound computation, stress-test regularization is fixed ($\alpha=100$, $\lambda=0.1$), not selected with held-out brand labels. Table~\ref{tab:lobo} reports unweighted brand-macro metrics. These are separate cross-brand stress tests, not the main 80/20 result.

\section{Results and Error Analysis}
Table~\ref{tab:main} shows that the CV-selected primary model is structural+similarity. Its test log RMSE is @@RMSE@@, with campaign-bootstrap 95\% interval [@@CILO@@, @@CIHI@@], and raw-view MAE @@RAWMAE@@. Structural-only subgraphs have RMSE @@STRUCT@@; added semantic links make only a small numerical difference. The paired log-MAE difference (similarity minus structural) has interval [@@DLO@@, @@DHI@@], including zero. Therefore we do not establish a reliable benefit from the extension.

The simpler video-node structural design achieves test RMSE @@NODE@@, lower than the primary subgraph model, although it was not selected by training CV. Metadata Ridge also remains competitive. We report these results directly: the richer graph does not establish superiority over a simpler representation or non-graph control. The constant baseline is weak; surpassing it alone would not justify claims about creative relationships.

\begin{figure}[t]\centering
\includegraphics[width=\columnwidth]{comparison.pdf}
\caption{Structural-only and similarity-augmented designs, compared with matched-information and metadata controls on the same 43 test videos.}\label{fig:comparison}
\end{figure}
\begin{table}[t]\caption{LOBO brand-macro stress test (14 held-out brands). These are refitted models under a separate protocol.}\label{tab:lobo}\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrr}\toprule
Model & Log MAE & Log RMSE & $\rho$\\\midrule
@@LROWS@@
\bottomrule\end{tabular}
\end{table}

Cross-brand RMSE remains substantially larger than the main test error for the primary GNN. Brand context, channel reach and curation differ; unseen categorical identities cannot be recovered by simply adding semantic links. LOBO also does not show the semantic extension reliably improving over structural graphs. Fine-grained brand estimates contain only about 14--20 test examples and should be interpreted cautiously.

Prediction scatter and exported brand/view-scale slices show compression of extremes and large individual failures. Paid promotion, targeting, channel audience, release timing and attention to featured people are plausible alternative explanations for views. Current channel metadata is a contemporaneous snapshot, not necessarily the channel audience at publication. We retain it for context rather than interpreting it as a pre-publication causal control. Source annotations may encode extractor style; short or duplicate evidence can dilute useful signals in mean pooling. Several source support edges and similarity links are heuristic, and weak textual supervision can spread irrelevant information.

\section{Evidence-Linked Recommendations and Showcase}
Recommendations follow: brief/objective $\rightarrow$ creative elements $\rightarrow$ predicted versus observed response with uncertainty $\rightarrow$ transferable hypothesis. Six cases are the first six fixed test indices, not the largest predicted gains. A frozen GNN searches training-observed (mechanic, structure, pacing) triples with at least two supporting examples and at most two categorical edits. It retains the original when no candidate improves predicted log views, and recomputes node text/similarity edges for edits. Brand, horizon, duration and existing textual evidence are held fixed, so a proposed categorical change may be inconsistent with the original script and requires human revision.

For one held-out @@CASEBRAND@@ case (source ID \texttt{@@CASEID@@}), the objective is to improve observable reach for the same product/context. Original observed views are @@CASEOBS@@; the model predicts @@CASEBASE@@ before and @@CASEBEST@@ after the proposed change: @@CASECHANGE@@. Training-only examples sharing the proposed triple are linked in the website. A nominal 90\% out-of-fold absolute-log-residual band gives [@@CASELOW@@, @@CASEHIGH@@] for the original prediction. This is a broad sensitivity band, not a guaranteed calibrated interval; it shows how uncertain the point prediction is. No edited-video response has been observed.

The new-video showcase uses a declared 30-second @@NEWBRAND@@ brief at an assumed 30-day exposure age. It builds a new graph from declared features/text only, without pretending that another video's evidence belongs to the new ad. Predicted views change from @@NEWBASE@@ to @@NEWBEST@@ under @@NEWCHANGE@@. The storyboard is: 0--3 s hook; 3--20 s demonstrate the product benefit; 20--27 s apply the suggested structure; 27--30 s CTA/end card. Full wording, supporting source links, sensitivity bands, and six existing-video cases accompany the report. This is a demonstration, not independent validation of actual uplift. Sparse new-brief graphs also differ from fully extracted training graphs.

The local Streamlit/Python prototype prioritizes feature-to-recommendation use, with editable settings, source evidence, the actual per-video GNN graph, structural/similarity display toggles, baseline comparisons, LOBO results and one storyboard. Python directly loads the frozen PyTorch Geometric checkpoint and training-fitted encoders for prediction and constrained search. It starts with \texttt{python -m streamlit run app.py}. Automated checks verify all 214 predictions against frozen outputs, all six optimization cases, graph controls and submitted Streamlit form interactions. Graph display switches do not change the trained prediction model.

\section{Critical Discussion and Improvements}
Higher views cannot be attributed to creative combinations in this observational sample. A celebrity's popularity/appearance, paid spend, platform distribution, audience size, campaign objectives and publication timing may jointly affect creative choices and responses. Likes are incomplete and comments are unavailable in this pass; neither brand recognition nor sales is measured. Temporal exposure is partly controlled through supplied dates, but dates and extracted concepts have not all been independently verified. Curated ad references, incomplete fresh-label coverage, imperfect campaign grouping, weak bilingual tokenization and a single outer split constrain generalization.

The recommendation search optimizes the model's estimate, which can exploit model error rather than improve an ad. Support from two training examples is a weak plausibility constraint, not causal identification. Uncertainty is large enough that favorable point estimates should remain hypotheses. A transferability claim should be restricted to the observed setting and supported creative elements; original copy and product claims should be rewritten and checked before use.

Priority improvements are: collect fixed post-publication horizons and media spend; verify campaign families and publication dates; expand the sample with a separately frozen official-channel holdout; hand-audit support links and test bilingual encoders; repeat grouped/time splits and capacity-matched graph ablations; and evaluate full sparse-brief versus extracted-graph prediction. Only a prospective exposure-controlled comparison of coherent video variants could test realized creative uplift. Such an experiment is not claimed here.

\section{Conclusion}
The supplied data supports a transparent compact GNN baseline with quantitative targets, a genuine grouped 80/20 evaluation, structural-edge ablations and cross-brand stress tests. The primary subgraph model predicts some popularity signal but does not establish superiority over simpler alternatives. The practical contribution is an auditable comparison and evidence-linked hypothesis tool whose limitations remain visible. The report uses the IEEE A4 conference template \cite{ieee}; reproducible code, frozen platform evidence, full metrics, model checkpoints and session transcript are provided separately.

\begin{thebibliography}{3}
\bibitem{sage} W. L. Hamilton, R. Ying, and J. Leskovec, ``Inductive representation learning on large graphs,'' in \emph{Advances in Neural Information Processing Systems}, vol. 30, 2017.
\bibitem{pyg} PyTorch Geometric, ``SAGEConv and graph-level pooling documentation.'' [Online]. Available: \url{https://pytorch-geometric.readthedocs.io/}
\bibitem{ieee} IEEE, ``Manuscript templates for conference proceedings.'' [Online]. Available: \url{https://www.ieee.org/conferences/publishing/templates.html}
\end{thebibliography}
\end{document}
'''
replacements={'N':len(json.loads((D/'raw_graphs.json').read_text())),'MAE':f(v['log_MAE']),'RMSE':f(v['log_RMSE']),'RHO':f(v['Spearman']),'NODE':f(m['test']['Node structural']['log_RMSE']),'GNODES':f'{ga["nodes"]:,}','GEDGES':f'{ga["provided_edges"]:,}','NGROUPS':a['groups'],'TRAIN':sp['train'],'TEST':sp['test'],'SEED':sp['seed'],'EPOCH':m['configs'][gnn]['epochs'],'WD':m['configs'][gnn]['weight_decay'],'ROWS':'\n'.join(rows),'LROWS':'\n'.join(lrows),'CILO':f(ci[0]),'CIHI':f(ci[1]),'RAWMAE':f'{v["views_MAE"]:,.0f}','STRUCT':f(m['test']['Subgraph structural']['log_RMSE']),'DLO':f(diff[0]),'DHI':f(diff[1]),'CASEBRAND':tex(case['brand']),'CASEID':tex(case['video_id']),'CASEOBS':f'{case["observed_views"]:,}','CASEBASE':f'{case["base_prediction"]:,.0f}','CASEBEST':f'{case["optimized_prediction"]:,.0f}','CASECHANGE':tex('; '.join(k.replace('_',' ')+' = '+v for k,v in case['changes'].items())),'CASELOW':f'{case["uncertainty_band"][0]:,.0f}','CASEHIGH':f'{case["uncertainty_band"][1]:,.0f}','NEWBRAND':tex(new['brand']),'NEWBASE':f'{new["base_prediction"]:,.0f}','NEWBEST':f'{new["optimized_prediction"]:,.0f}','NEWCHANGE':tex('; '.join(k.replace('_',' ')+' = '+v for k,v in new['changes'].items()) or 'unchanged supported settings')}
for k,val in replacements.items():source=source.replace('@@'+k+'@@',str(val))
assert '@@' not in source
out=R/'output/latex';out.mkdir(exist_ok=True)
(out/'Group6_Final_Report.tex').write_text(source)
for name in ['comparison.pdf','scatter.pdf']:shutil.copy2(D/name,out/name)
shutil.copy2(R/'references/ieee/IEEEtran.cls',out/'IEEEtran.cls')
print('IEEE A4 source:',out/'Group6_Final_Report.tex')

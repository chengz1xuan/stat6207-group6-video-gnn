"""Generate the four-page v4 IEEE report from frozen audited/paired results."""
from pathlib import Path
import json,shutil,csv
R=Path(__file__).resolve().parents[1]
load=lambda p:json.loads(p.read_text())
config=load(R/'artifacts/final_v4.json');D=R/config['primary_folder'];P=R/config['paired_directory']
m=load(D/'metrics.json');w=load(D/'web_results.json');o=load(D/'optimization.json');u=load(D/'uncertainty.json');diag=load(D/'report_diagnostics.json');paired=load(P/'comparison.json');authors=load(R/'AUTHORS.json')
primary=m['selected_primary_gnn'];v=m['test'][primary];ages=w['age_support'];edge=w['brand_edge_audit'];case=o['dataset_cases'][0];brief=load(R/config['showcase_path']);ga=w['graph_audit'];worst=w['largest_error'];graph=diag['display_graph'];cp=diag['paired_RMSE_comparisons']
def tex(s):return ''.join({'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','\\':r'\textbackslash{}'}.get(c,c) for c in str(s))
def f(x):return '--' if x is None else f'{x:.3f}'
def count(x):return f'{x:,.0f}'
labels={'Mean':'Constant log mean','Brand mean':'Brand log mean','Metadata Ridge':'Metadata Ridge','Pooled Ridge':'Pooled Ridge','Pooled MLP':'No-message MLP','Subgraph structural':'Subgraph: core','Subgraph + similarity':'Subgraph: core+sim.','Node structural':'Video: brand cliques','Node + similarity':'Video: cliques+sim.','Node no-brand edges':'Video: no brand edges'}
rows=[]
for n,z in m['test'].items():
    label=labels[n]+('*' if n==primary else '')
    rows.append(' & '.join([label,f(m['CV_log_RMSE'].get(n)),f(z['log_MAE']),f(z['log_RMSE']),f(z['Spearman'])])+r' \\')
lrows=[' & '.join([labels[n],f(z['macro_log_MAE']),f(z['macro_log_RMSE']),f(z['macro_Spearman'])])+r' \\' for n,z in w['lobo'].items()]
prows=[]
for name in ['Subgraph structural','Subgraph + similarity']:
    z=paired['model_comparisons_seed42'][name]
    prows.append(' & '.join([labels[name],f(z['v2_log_RMSE']),f(z['v3_log_RMSE']),f'[{z["delta_log_RMSE_95"][0]:.3f}, {z["delta_log_RMSE_95"][1]:.3f}]'])+r' \\')
cv=m['configs'][primary];ci=u['test_log_RMSE_95'][primary];fields=['structure_type','pacing'];oldcase='; '.join(k.replace('_',' ')+': '+case['creative_elements'][k]+' to '+case['changes'][k] for k in fields if k in case['changes']);newchange='; '.join(k.replace('_',' ')+': '+brief['row'][k]+' to '+val for k,val in brief['changes'].items())
prepared_rows=list(csv.DictReader((R/config['prepared_directory']/'videos.csv').open()))
byyoutube={r['youtube_id']:r for r in prepared_rows};registry=load(R/config['prepared_directory']/'review_rules.json')
family_labels={'ikea-complete-sleep':'IKEA Complete Sleep','mcdonalds-shogun-tamago':"McDonald's Shogun",'prudential-170-anniversary-2019':'Prudential 170th','mannings-everyday-wellness-2023':'Mannings wellness', 'mcdonalds-world-cup-wow-sauce-2026':"McDonald's WOW sauce",'ikea-home-storage-for-love':'IKEA storage','ikea-tested-quality-proposal':'IKEA Proposal','maxims-blessings-cny-2023':"Maxim's CNY 2023"}
family_rows=[]
for family in registry['campaign_families']:
 if family['status']!='text_evidence':continue
 roles={byyoutube[k]['split'] for k in family['youtube_ids']};assert len(roles)==1
 family_rows.append(' & '.join([tex(family_labels[family['campaign_id']]),str(len(family['youtube_ids'])),next(iter(roles))])+r' \\')
family_rows.append(r'Five unresolved families & 13 & quarantine \\')
source=r'''\documentclass[a4paper,conference]{IEEEtran}
\usepackage[T1]{fontenc} % Actual Times regular/bold/italic fonts under XeTeX.
\usepackage{amsmath,graphicx,booktabs,url,balance}
\usepackage[hidelinks]{hyperref}
\title{Audited Advertising Graphs for Video Views:\\A Paired Policy Evaluation}
\author{\IEEEauthorblockN{@@NAME0@@}
\IEEEauthorblockA{The Chinese University of Hong Kong\\Hong Kong SAR, China\\@@EMAIL0@@}
\and
\IEEEauthorblockN{@@NAME1@@}
\IEEEauthorblockA{The Chinese University of Hong Kong\\Hong Kong SAR, China\\@@EMAIL1@@}
\and[\hfill\mbox{}\par\medskip\mbox{}\hfill]
\IEEEauthorblockN{@@NAME2@@}
\IEEEauthorblockA{The Chinese University of Hong Kong\\Hong Kong SAR, China\\@@EMAIL2@@}
\and
\IEEEauthorblockN{@@NAME3@@}
\IEEEauthorblockA{The Chinese University of Hong Kong\\Hong Kong SAR, China\\@@EMAIL3@@}}
\begin{document}\maketitle
\begin{abstract}
We assess graph neural networks (GNNs) for advertising-video cumulative views. A frozen YouTube collection provides 214 exact labels across 14 brands. A source-text audit merges campaign variants, including previously separated IKEA and McDonald's series and Maxim's New Year chapters; unresolved families are quarantined. The revised evaluation uses 161 training, 40 testing and 13 quarantined videos. All models share the grouped split and training-only tuning. The selected subgraph GNN reaches log RMSE @@RMSE@@, versus @@META@@ for metadata Ridge. A paired old/new text-and-graph-policy comparison holds cohort, labels, folds and tuning budgets fixed; its intervals do not establish improvement. Video-node adjacency is entirely same-brand connectivity. A local Streamlit tool shows real graphs, source-linked cases and an author-declared storyboard. Results are post-audit internal evidence; neither causal uplift nor 30-day forecasts are validated.
\end{abstract}
\begin{IEEEkeywords}Graph neural networks, cumulative views, campaign grouping, brand confounding, paired evaluation.\end{IEEEkeywords}

\section{Question, Data and Evaluation Version}
We ask whether creative graph structure predicts observable audience response beyond metadata and pooled features. The target is $y=\ln(1+v)$ for views $v$ at capture, not sales or brand recognition. Following the supplied slide-level specification, we implement a compact GNN rather than claim reproduction of unavailable VideoGNN code or an unidentified 120-video subset.

Uniform latest-version-per-brand selection yields 219 unique URLs; HSBC/Hang Seng use v0.2 and MTR contains 14 rows. Archived counts are mainly JSON \texttt{engagement\_stats} in each package's evidence table (219 views, 184 likes, no comments). A public unauthenticated collection on 9 November 2026, 12:01:03--12:02:18 UTC, recovers 214 exact current-video view counts/channel identities and 175 exact likes. Five unavailable fresh labels are excluded without archived substitution. Missing comments are not zero. URL/ID, timestamps, status, response digests and extraction snippets preserve provenance; rounded subscriber strings are not model inputs.

This paper reports \emph{v4}, superseding the earlier 171/43 report for this branch. Source-supported parent-series rules now merge IKEA Complete Sleep, McDonald's Shogun Tamago/WOW sauce and Maxim's 2023 New Year chapters, among others. For example, \texttt{CxbnjaET7ew} and \texttt{WBMGLbAvnQY} share the named New Year series and 29 December 2022 release date despite different product labels. Existing groups are only unioned, never split. Thirteen registry families comprise eight text-supported families and five unresolved families; unresolved connected components (13 videos) are excluded from fitting and scoring. Text support establishes a conservative campaign boundary, not verified identical footage. Other creative annotations still require review.

The eligible cohort has @@GROUPS@@ groups over 201 videos. A fixed seed-42 whole-group subset-sum rule chooses the closest rounded 20\% count without labels: 161 training/40 testing (80.10/19.90\%). Every registered family remains within one outer partition and one role in each training CV fold. Thus known cross-boundary variants are repaired, but exhaustive campaign lineage is not claimed. The holdout was already inspected during prior audits; this is corrected internal evaluation, not a new untouched external test. No fresh label was altered to improve scores.

Publication dates/durations remain source-supplied, not newly verified platform dates. Training age ranges from @@AGEMIN@@ to @@AGEMAX@@ days (median @@AGEMED@@). Heterogeneous-age cumulative labels do not identify a response trajectory. Thirty days is outside training support; an in-range age also does not validate a new-video time forecast.

\section{Graphs, Features and Baselines}
\subsection{Video as a Creative/Evidence Subgraph}
The primary representation is one graph per video: a context root, creative aspects and observed-text evidence. It retains original \texttt{candidate\_edges} memberships and deterministic observation-support links, with original IDs. It excludes response statistics, assessment/risk scores, derived-summary evidence and unreviewed heuristic support links. Quantified-response sentences are removed while ordinary uses of ``like''/``share'' survive. Duplicate-summary hooks are blanked and duplicated value-promise/proof text is omitted, without inventing replacement observations. Caps are three evidence nodes per kind, one aspect per field and four visual/audience aspects. Eligible graphs contain @@NODES@@ nodes and @@EDGES@@ original relations, including @@SUPPORTS@@ retained support links.

The root encodes brand/seven creative categories, log-duration/age with missingness flags and 16-dimensional TF-IDF/SVD message/hook/promise/CTA text. Nodes add type/field indicators and a separate 16-dimensional text representation. All encoders, medians, scalers, vocabularies and SVD fit training data only. The semantic extension adds within-video nonroot top-two neighbors with training-fitted SVD cosine $\geq0.55$, then symmetrizes/deduplicates. These links are explicitly heuristic. Figure~\ref{fig:graph} shows actual source and semantic relationships.

\begin{figure}[t]\centering\includegraphics[width=\columnwidth]{v4_graph.pdf}
\caption{Real training graph \texttt{@@GRAPHYT@@}, displaying @@SHOWCOUNT@@ of @@FULLCOUNT@@ nodes. Gray: source membership; blue: retained source concept--evidence support; orange dashed: actual heuristic semantic extension. Nodes/edges and original text are exported in \texttt{report\_diagnostics.json}. Selection uses graph structure, never response labels.}\label{fig:graph}\end{figure}

Two 24-unit mean GraphSAGE layers \cite{sage,pyg,sageconv} use
\begin{equation}
h_u^{\ell+1}=\operatorname{ReLU}\!\left(W_s h_u^\ell+W_n\operatorname{mean}_{w\in N(u)}h_w^\ell+b\right).
\end{equation}
A linear head on concatenated root/mean embeddings predicts standardized log views. Adam (learning rate 0.01) minimizes MSE. Controls are constant training log mean; training brand log mean with global-mean fallback for unseen brands; metadata Ridge (brand/duration/age); pooled-root/mean-feature Ridge; and a two-layer no-message MLP with the same information/readout but a different parameter count. All predict the same target.

\subsection{Video Nodes and Brand Adjacency}
The extension makes each video a node with root features. Supplied same-brand/author/creative relations collapse to @@PAIRS@@ binary pairs, exactly complete brand cliques. Creative relation labels add no independent adjacency. A no-brand-edge control removes every same-brand endpoint pair, leaving zero edges while retaining brand features, architecture and tuning budget. A separate extension adds five nearest training-feature neighbors. Queries receive messages only from training references, with no query-to-training/query messages. Clique performance cannot demonstrate effective concept relations.

\section{Training, Uncertainty and Paired Policies}
Three grouped folds within the 161 training videos select Ridge $\alpha\in\{10,100\}$ and neural weight decay $\lambda\in\{0.01,0.1\}$. Each fold refits preprocessing; validation log MSE stops training within 260 epochs (patience 35). Mean fold log RMSE selects parameters, and median selected-fold epoch determines final refitting on all training rows. Quantitative training/CV/test measurements are exported for all ten models.

The prespecified primary family is subgraph-as-video. CV selects core+similarity: @@CVSIM@@ versus @@CVCORE@@ for core-only, despite core-only's better test RMSE. Final primary training uses @@EPOCH@@ epochs, $\lambda=@@WD@@$, training log RMSE @@TRAINRMSE@@. Across the whole suite, CV prefers the video-node clique model (@@CVNODE@@); it does not redefine the primary design. Test rankings never change model selection.

\begin{table}[t]\caption{Same v4 cohort. CV is training-only tuning error; other columns are held-out results ($n=40$). * marks the primary subgraph selected by CV. Errors use natural-log views.}\label{tab:main}\centering\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrr}\toprule
Model & CV RMSE & Test MAE & Test RMSE & $\rho$\\\midrule
@@ROWS@@
\bottomrule\end{tabular}\end{table}

Common metrics are log MAE/RMSE and Spearman correlation. Back-transformed $\exp(\hat y)-1$ is clipped nonnegative, not an unbiased arithmetic expected count. Campaign bootstrap (2,000 paired draws, seed 44) gives conditional intervals, excluding split/selection uncertainty. Primary test RMSE @@RMSE@@ has 95\% interval [@@CILO@@, @@CIHI@@]. Its RMSE difference versus metadata Ridge has interval [@@MDLO@@, @@MDHI@@], spanning zero. Node clique-minus-no-edge difference also spans zero [@@NDLO@@, @@NDHI@@]. We establish no reliable graph advantage.

\subsection{Matched Old/New Policy Experiment}
We retrain both legacy and revised policies on the \emph{same v4 eligible videos, labels, groups, outer split, inner folds, video-node pairs, architectures and tuning budget}. The legacy arm restores original text and frozen old subgraphs; the revised arm uses cleaned text and rebuilt conservative graphs. Shared mean/brand-mean/metadata predictions must agree (absolute tolerance $10^{-6}$), otherwise comparison fails. Both subgraph architectures use seeds 42, 7, 21 and 84; only seed 42 tunes, with other seeds using selected regularization/epochs. Table~\ref{tab:paired} reports seed-42 paired RMSE; all seed predictions/checkpoints are retained.

\begin{table}[t]\caption{Matched-policy test RMSE and 95\% campaign-bootstrap difference interval (revised minus legacy). Negative favors revised. Neither interval establishes improvement.}\label{tab:paired}\centering\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrr}\toprule
Subgraph & Legacy & Revised & Difference interval\\\midrule
@@PROWS@@
\bottomrule\end{tabular}\end{table}

These intervals describe the combined text/graph policy and fitted learning procedure; they do not isolate removal of one hook or edge, or identify causal creative effects. Earlier 171/43 scores are not compared directly with v4 as if paired. Additional seeds probe initialization; no favorable test seed replaces the primary.

\section{Errors and Cross-Brand Transfer}
Figure~\ref{fig:scatter} includes all 40 test videos. The largest primary log error is \texttt{@@WORSTYT@@} (Caf\'e de Coral): @@WORSTOBS@@ observed versus @@WORSTPRED@@ predicted, despite an official uploader. The historical third-party McDonald's archive \texttt{P25f\_dDd2MI} had 1,019 views versus about 442,536 under the superseded model; that is historical evidence, not a v4 score. Its 1996 creative year differs from its supplied 2026 upload date. Channel identity, distribution and paid exposure may differ, but their causal contributions are not measured.

\begin{figure}[t]\centering\includegraphics[width=\columnwidth]{v4_scatter.pdf}
\caption{All 40 v4 test predictions: primary GNN (green) and metadata Ridge (gray crosses). Dashed line: perfect prediction; circle: largest primary log error. No training predictions enter this plot.}\label{fig:scatter}\end{figure}

LOBO refits on 13 brands, holding out each of 14 in turn. Training-only grouped inner stopping is followed by refitting; separate fixed regularization is $\alpha=100$, $\lambda=0.1$. Table~\ref{tab:lobo} reports five-model brand-macro metrics (70 runs). Brand mean falls back to global training mean and has undefined within-brand rank correlation. These stress tests are separate from the main 80/20 result.

\begin{table}[t]\caption{V4 leave-one-brand-out macro results (14 brands).}\label{tab:lobo}\centering\scriptsize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrr}\toprule Model & Log MAE & Log RMSE & $\rho$\\\midrule
@@LROWS@@
\bottomrule\end{tabular}\end{table}

\section{Source-Linked Cases and Demonstrator}
Six existing-video cases are the first six fixed test indices. Frozen-GNN search changes at most two mechanic/structure/pacing categories, requiring each proposed triple to occur at least twice in training. It retains the original when no candidate improves the estimate. Extracted evidence stays fixed; categories must be reconciled with a rewritten script before use. Nominal 90\% out-of-fold absolute-log-residual ranges are sensitivity summaries, not calibrated individual intervals or uplift uncertainty.

\emph{Existing brief:} Hang Seng credit-card payment convenience, source \texttt{@@CASEYT@@}, with observed @@CASEOBS@@ views. The proposed edits are @@CASECHANGE@@. Original prediction @@CASEBASE@@ becomes @@CASEBEST@@ over @@CASECAND@@ candidates; its original sensitivity range is [@@CASELOW@@, @@CASEHIGH@@]. Supporting training examples include recipe-app \texttt{@@SUPPORT0@@} and IKEA storage \texttt{@@SUPPORT1@@}; they support a categorical triple, not guaranteed cross-product transfer. Takeaway: test a coherent problem/solution edit under comparable exposure; no edited outcome is observed.

\emph{New author-declared brief:} a hypothetical 30-second MTR route-planning ad with hook ``Planning a cross-boundary journey? Start with a clear route.'' Age is the training median @@AGEMED@@ days, not a 30-day forecast. Proposed edits: @@NEWCHANGE@@. Predicted cumulative views change from @@NEWBASE@@ to @@NEWBEST@@; original sensitivity range [@@NEWLOW@@, @@NEWHIGH@@]. A graph contains declared inputs only, with no borrowed extracted evidence. Storyboard: 0--3 s question; 3--20 s travel need/route-map option; 20--27 s proposed structure/pacing; 27--30 s check-current-timetable CTA. Wording is proposed, not verified current service information; there is no observed response. Training support and model errors make the recommendation a hypothesis.

The Streamlit/Python prototype defaults to this v4 checkpoint, with source graphs, complete cases, matched experiment dictionary/metrics and the report download. Historical results remain selectable and explicitly labeled. Changing versions clears stale inputs/predictions. Out-of-support ages, including 30 days, trigger warnings in the UI and JSON; in-range ages and sparse new-brief graphs are still unvalidated. ZIP packaging includes the runtime policy module and frozen input/result bundles.

\section{Reproducibility and Audit Checks}
The frozen branch records preparation rules/inputs, fold identities, code hashes, paired controls, checkpoints and environment versions. Fifty-seven unit tests pass on macOS in Conda \texttt{stat6207}; all 201 revised and 214 historical predictions match saved outputs within $10^{-4}$ log units. App tests cover age warnings, optimization and switching. The registered families cover 32 of 214 source videos (Table~\ref{tab:families}); this targeted registry is not an exhaustive video-level campaign audit. Both source-backed known unions and unresolved-family quarantine are checked through outer/inner partitions. Package verification includes importing the extracted App dependencies and checking result hashes; these checks do not establish predictive or causal validity.

\begin{table}[t]\caption{Registered family components in v4. Each supported family has one outer role; unresolved connected families are excluded entirely.}\label{tab:families}\centering\scriptsize\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrl}\toprule Family & Videos & V4 role\\\midrule
@@FAMILIES@@
\bottomrule\end{tabular}\end{table}

\section{Limitations and Conclusion}
Paid media, audience targeting, uploader reach, timing, reupload status and featured people's popularity may confound creative choices and views. Contemporary channel metadata is not historical exposure. Source dates, bilingual tokenization, heuristic semantic links, annotation noise, unresolved creative reviews, non-exhaustive campaign coverage and an inspected single holdout limit validity. Search can exploit model error; two supporting examples cannot establish actual uplift.

Priority improvements are verified campaign/uploader/date review, fixed-age response/spend collection, an independently frozen external holdout, repeated grouped/time splits and exposure-controlled coherent variants. The revised GNN baseline is quantitatively reproducible, but neither graph superiority nor a policy-cleaning gain is established. Cumulative snapshot estimates must remain separate from validated future forecasts. The contribution is an auditable comparison and source-linked hypothesis interface, with no claim of sales, brand recognition or causal engagement gains. This four-page IEEE A4 report \cite{ieee} accompanies code, frozen evidence, checkpoints and the visible session transcript.

\balance
\begin{thebibliography}{4}
\bibitem{sage} W. L. Hamilton, R. Ying, and J. Leskovec, ``Inductive representation learning on large graphs,'' in \emph{Advances in Neural Information Processing Systems}, vol. 30, 2017, pp. 1024--1034. [Online]. Available: \url{https://papers.nips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html}
\bibitem{pyg} M. Fey and J. E. Lenssen, ``Fast graph representation learning with PyTorch Geometric,'' in \emph{ICLR Workshop on Representation Learning on Graphs and Manifolds}, 2019, arXiv:1903.02428. [Online]. Available: \url{https://arxiv.org/abs/1903.02428}
\bibitem{sageconv} PyTorch Geometric, ``torch\_geometric.nn.conv.SAGEConv,'' \emph{Official API documentation}. Accessed: Nov. 13, 2026. [Online]. Available: \url{https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.SAGEConv.html}
\bibitem{ieee} IEEE, ``Manuscript templates for conference proceedings.'' Accessed: Nov. 13, 2026. [Online]. Available: \url{https://www.ieee.org/conferences/publishing/templates.html}
\end{thebibliography}\end{document}
'''
rowsdf=w['videos'];byid={r['video_id']:r for r in rowsdf};ci_meta=cp[primary+' minus Metadata Ridge']['campaign_bootstrap_95'];ci_node=cp['Node structural minus Node no-brand edges']['campaign_bootstrap_95'];supports=case['support_examples'];lastsupport=next((r for r in supports if 'IKEA' in r['title']),supports[-1]);YT=lambda sid:byid[sid]['youtube_id']
repl={'FAMILIES':'\n'.join(family_rows),'NAMES':r'\quad '.join(tex(a['name']) for a in authors),'RMSE':f(v['log_RMSE']),'META':f(m['test']['Metadata Ridge']['log_RMSE']),
 'GROUPS':len({r['group'] for r in rowsdf}),'AGEMIN':f'{ages["training_min_days"]:.1f}','AGEMAX':f'{ages["training_max_days"]:.1f}','AGEMED':f'{ages["training_median_days"]:.1f}',
 'NODES':f'{ga["nodes"]:,}','EDGES':f'{ga["provided_edges"]:,}','SUPPORTS':sum(e['type']=='aspect_supported_by_evidence' for g in w['graphs'] for e in g['edges']),
 'GRAPHYT':tex(graph['youtube_id']),'SHOWCOUNT':len(graph['nodes']),'FULLCOUNT':graph['full_node_count'],'PAIRS':f'{edge["core_undirected_pairs"]:,}',
 'CVSIM':f(m['CV_log_RMSE']['Subgraph + similarity']),'CVCORE':f(m['CV_log_RMSE']['Subgraph structural']),'EPOCH':cv['epochs'],'WD':cv['weight_decay'],'TRAINRMSE':f(cv['train_log_RMSE']),
 'CVNODE':f(m['CV_log_RMSE']['Node structural']),'ROWS':'\n'.join(rows),'PROWS':'\n'.join(prows),'LROWS':'\n'.join(lrows),'CILO':f(ci[0]),'CIHI':f(ci[1]),
 'MDLO':f(ci_meta[0]),'MDHI':f(ci_meta[1]),'NDLO':f(ci_node[0]),'NDHI':f(ci_node[1]),'WORSTYT':tex(worst['youtube_id']),'WORSTOBS':count(worst['views']),'WORSTPRED':count(worst['predicted_views']),
 'CASEYT':tex(YT(case['video_id'])),'CASEOBS':count(case['observed_views']),'CASECHANGE':tex(oldcase),'CASEBASE':count(case['base_prediction']),'CASEBEST':count(case['optimized_prediction']),
 'CASECAND':case['candidates_evaluated'],'CASELOW':count(case['uncertainty_band'][0]),'CASEHIGH':count(case['uncertainty_band'][1]),'SUPPORT0':tex(YT(supports[0]['video_id'])),'SUPPORT1':tex(YT(lastsupport['video_id'])),
 'NEWCHANGE':tex(newchange),'NEWBASE':count(brief['base_prediction']),'NEWBEST':count(brief['optimized_prediction']),'NEWLOW':count(brief['uncertainty_band'][0]),'NEWHIGH':count(brief['uncertainty_band'][1])}
repl.update({'EMAIL'+str(i):tex(a['email']) for i,a in enumerate(authors)})
repl.update({'NAME'+str(i):tex(a.get('ieee_name',a['name'])) for i,a in enumerate(authors)})
for key,value in repl.items():source=source.replace('@@'+key+'@@',str(value))
assert '@@' not in source
out=R/'output/latex';out.mkdir(exist_ok=True)
(out/'Group6_Final_Report.tex').write_text(source,encoding='utf-8')
for before,after in [('report_graph.pdf','v4_graph.pdf'),('report_scatter.pdf','v4_scatter.pdf')]:shutil.copy2(D/before,out/after)
shutil.copy2(R/'references/ieee/IEEEtran.cls',out/'IEEEtran.cls')
print('Wrote v4 report source with four named authors and measured audited/paired results.')

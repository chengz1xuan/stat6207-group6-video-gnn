Hi Ming — I prepared another controlled Mplifier graph-probe pack, this time using 16 HSBC-family ads.

I checked the set before building it. All 16 are retained as HSBC-family material. 15 are from the HSBC Hong Kong channel; one is a 1995 HSBC ad from a third-party Hong Kong TV-ad archive. A few UI-level brand fields are noisy, so for the probe I used the canonical brand/evidence and normalized HSBC / HSBC Life under the parent HSBC family while still preserving the sub-brand.

The pack has 16 video nodes, 704 aspect nodes, 1037 evidence nodes and 3554 candidate edges.

Same caveat as before: the edge table is not ground truth. Structural links and same-brand / same-author / same-mechanic / same-structure / same-evidence-mode type relationships are deterministic under the stated rules. `similar_aspect_candidate` is deliberately weaker and is only a similarity proposal for review.

I’m still treating this as a graph-readiness / clustering exercise, not a claim that Mplifier already has a GNN. I think this HSBC set is useful because it is one parent brand but spans banking/app utility, wealth/Premier, HSBC Life/insurance and brand-level work — so we can see whether useful sub-clusters emerge inside the same brand.

Compared with the previous 20-case McDonald’s pack, this one is smaller in number of videos but more heterogeneous within the parent brand. I’d be interested in your view on which node/edge types are actually useful, which are noise, and what you would want to change before any graph-embedding or future GNN-style test.

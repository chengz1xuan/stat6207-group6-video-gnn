Hi Ming — I prepared another controlled Mplifier graph-probe pack, this time using 20 Hang Seng Bank ads.

I checked the set first. All 20 are retained and all 20 come from the official 恒生銀行 Hang Seng Bank YouTube channel. The canonical brand labels vary slightly in capitalization / Chinese-English format, so I normalized them to one parent `Hang Seng Bank` family while preserving the original brand value.

The pack has 20 video nodes, 794 aspect nodes, 1244 evidence nodes and 4703 candidate edges.

Same caveat: the edge table is not ground truth. Structural and exact same-field relationships are deterministic under the stated rules; `similar_aspect_candidate` is deliberately weaker and should only be treated as a review proposal.

I think this set is a useful next test after McDonald’s and HSBC because brand and author are almost completely controlled here — same parent brand, same official channel — but the campaigns span different products and creative approaches such as credit cards, +FUN Dollars, travel, banking tiers, offers and celebrity-led executions.

So the more interesting question is whether Mplifier’s creative fields can separate useful sub-clusters even when same-brand and same-author edges have almost no discriminative value.

I’d be interested in which fields you think are carrying real cluster signal, which edges should be dropped as saturated/noisy, and what you would want before trying graph embeddings or any later GNN-style experiment.

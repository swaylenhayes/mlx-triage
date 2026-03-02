# How the Interactive Application Project Connects to mlx-triage as Evidence

Reading the experiment consolidation in the context of the C1/C2 evidence, the connection is much tighter than "related topics" — the experiment consolidation is **primary evidence for the exact gap mlx-triage is designed to fill.**

The entire "Infrastructure Illusions" section — the four-cycle feedback loop — is a lived case study of the problem the research literature describes in abstract. The project went from 2 working models to 10 usable models without changing a single weight. The C1 evidence catalogs the *mechanisms* (EOS mishandling, template fallback, KV cache bugs, batch invariance violations). The write-up documents the *practitioner experience* of stumbling through those mechanisms without a diagnostic framework to tell which layer was broken.

The most powerful alignment is here: Experiment 6 is essentially a manual, intuition-driven version of what mlx-triage would automate. Hit a wall, hypothesized whether it was model or infrastructure, wrote a spec, got a fix, and validated. Four times. Each cycle took days. The tool would compress that to minutes by running the tiered protocol and telling the developer upfront: "this is an EOS handling issue" or "this is a model capacity issue — no engine fix will help."

Three specific findings from the write-up are directly citable as evidence for the tool's value proposition:

**The WaveCut story** is the headline case study. 333 tok/s, understood tool calls, but the server couldn't parse its output format. It was documented as "eliminated." It took Cycle 1's engine fix to reveal it was the fastest model in the roster. A Tier 0 tokenizer/EOS check would have flagged this immediately — it's exactly the class of issue Check 0.2 is designed to catch.

**VLM degeneration is the counter-example that proves the diagnostic framework needs both directions.** The experiment tested 7 configurations of frequency_penalty, tried OCR augmentation, tried cloud hosting — and confirmed it's a parameter-scale problem, not infrastructure. This is the "try better weights first" conclusion from C2, validated empirically. The tool needs to be able to say "this is NOT infrastructure, stop debugging the engine" just as clearly as it says "this IS infrastructure."

**The gemma-3n elimination** — 1,709 missing AltUp parameters in mlx_vlm — is a Tier 0 weight integrity issue. The tool's Check 0.3 (statistical sanity on weight tensors, checking for missing/zero layers) would catch this pattern: structurally incomplete weights that pass basic loading but produce garbage output.

The four-cycle feedback loop maps onto the three-layered conclusion from the synthesis: Cycle 4 (cloud comparison, no wall) is Regime 1 (steady-state, model quality is the binding constraint). Cycles 1-3 (infrastructure walls) are Regime 2 (setup/configuration, infrastructure failures common and routinely misattributed). And the fact that there was no framework for knowing which regime was active at any given moment is Regime 3 — the uncharacterized boundary that mlx-triage is designed to map.

This is the strongest piece of evidence because it's first-party — the experiments were run, the failures were documented, and the before/after data exists. The C1/C2 research gives the academic and industry backing. The experiment consolidation gives the practitioner story. Together they make the case that this tool needs to exist.

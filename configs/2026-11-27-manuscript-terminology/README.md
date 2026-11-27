# Terminology in the manuscript and public instructions

The main manuscript has 15 pages, with the references beginning on page 9.
Section 3 introduces the settings. Sections 4, 5 and 6 cover ASIS, CTRL and
Unified and Extended evaluation respectively. Section 7 covers representation
geometry. The manuscript does not label these experiments Step 1 through Step 4.

The audit searched the 165 tracked Markdown files in the parent commit and
reviewed the public navigation, method instructions, protocol names and dated
records separately. In particular, the earlier VSSL recipe collection is a
subset of the historical model configurations. Figure 3 compares frozen
readouts and LoRA; full fine-tuning results are tabulated in the supplement.
Figure 4 changes multiple conditions between ASIS and CTRL.

Executable configuration names and Appendix E filenames retain their spelling.
Archived protocols, upstream sources and recorded results retain their original
identities. The current figure guide documents the disagreement between the
Section 6 description of RAEv2 LP/AP and the recovered tokenizer runs. The
two-feature plane and the separate reference-free analysis in Section 7 also
have different inputs; the recovered reference-free fits used three or four
coordinates across folds. These documentation changes do not select a different
model, change measurements or alter training code.

The focused tests failed on the old section correspondence, numbered method
instructions, ambiguous figure labels and controlled-recipe checks before the
edits. Run the tests with Python 3.12:

```sh
python3 -m unittest tests.test_paper_terminology tests.test_current_doc_terms tests.test_public_manuscript_language tests.test_method_manuscript_terms tests.test_step2_docs_consistency tests.test_paper_guide_navigation tests.test_reviewer_protocol_index
python3 bin/mutate.py --spec mutations/paper-terminology.json
```

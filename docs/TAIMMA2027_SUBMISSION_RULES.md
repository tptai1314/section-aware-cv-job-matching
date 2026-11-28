# TAIMMA 2027 Submission Rules & Springer LNCS Template Guide

This document archives the official formatting, review, and submission rules for **TAIMMA 2027** (Springer LNCS), paired with the official template stored in [`templates/springer-lncs/`](../templates/springer-lncs/).

## 1. Originality & Format Requirements
- **Originality**: Papers must be original, unpublished, and not under review elsewhere.
- **Template**: Prepared strictly in the **Springer LNCS format** (`templates/springer-lncs/llncs.cls`, `splncs04.bst`, Version 2.25 of 2026/09/03).
- **Page Limits**:
  - **Full papers**: Minimum **12 pages**, Maximum **15 pages**.
  - **Systematic literature reviews / survey papers**: Up to **17 pages** in Springer format without additional page fee.
  - **Short papers**: Minimum **6 pages**.

## 2. Figures, Tables & Layout Constraints
- **No AI-generated images or diagrams**: AI-generated images and diagrams should be avoided. Use native vector graphics (TikZ) or genuine empirical plots (e.g., Matplotlib/EPS/PDF).
- **Text area compliance**: All figures, tables, and text must remain strictly within the page text area (no overfull margins).
- **Cross-referencing**: Every figure and table must be explicitly referred to in the paper text.

## 3. Single-Blind Peer Review & Authorship
- **Single-blind review**: Authors **must include** their full names, affiliations (including complete institutional addresses), and email addresses in the submitted manuscript.
- **Evaluation**: Each paper is assigned to at least 3 reviewers (evaluated for originality, relevance, methodological/technical quality, clarity, and contribution). Plagiarism and textual-overlap checking will be conducted.
- **Submission Portal**: Electronic submission via [Microsoft CMT](https://cmt3.research.microsoft.com/TAIMMA2027).
- **Code of Conduct**: All authors must follow the [Springer Nature Code of Conduct for Book Authors](https://www.springernature.com/gp/authors/book-authors-code-of-conduct).

## 4. Required Acknowledgments
- Include the **Competing Interests** declaration (`\subsubsection{\discintname}`) inside `\begin{credits} ... \end{credits}` per LNCS v2.25.
- Microsoft CMT acknowledgment:
  > *"The Microsoft CMT service was used for managing the peer-reviewing process for this conference. This service was provided for free by Microsoft and they bore all expenses, including costs for Azure cloud services as well as for software development and support."*

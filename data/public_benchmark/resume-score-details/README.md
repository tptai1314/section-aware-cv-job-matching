---
license: cc
task_categories:
- text-classification
- feature-extraction
language:
- en
tags:
- hr
size_categories:
- 1K<n<10K
---


# Resume and Job Description Matching Dataset

### Overview
This dataset contains **1,031 samples** of resumes and job descriptions (JDs) generated and assessed using **GPT-4o**. The primary goal of this dataset is to evaluate the alignment between resumes and job descriptions, aiding in the study of resume relevance, skill alignment, and job fit scoring based on predefined criteria.

### Dataset Composition
The dataset includes resumes matched with job descriptions, with the assessment and scoring details based on various matching criteria:
- **201 Mismatched JSONs**: Resumes that are not relevant to the provided JD.
- **648 Matched JSONs**: Resumes that are relevant and aligned with the JD.
- **142 Invalid JSONs**: Cases where either the resume or JD is incomplete or invalid.
- **40 JSONs Missing Additional Info**: Instances where additional input information was omitted.

### Dataset Structure
Each sample JSON file in the dataset includes the following keys:
- **`input`**:
  - **`job_description`**: Contains the full job description text.
  - **`macro_dict`**: A dictionary with macro-level criteria and their respective weighting.
  - **`micro_dict`**: A dictionary with micro-level criteria and their respective weighting.
  - **`additional_info`**: Extra requirements or preferences related to the JD.
  - **`minimum_requirements`**: List of fundamental qualifications for the role.
  - **`resume`**: Text of the resume as provided.

- **`output`**:
  - **`justification`**: Reasons for the scores assigned, based on specific criteria.
  - **`scores`**:
    - **`macro_scores`**: Scores for broader criteria (e.g., experience, strategic thinking).
    - **`micro_scores`**: Scores for detailed criteria (e.g., market research expertise).
    - **`requirements`**: Boolean indicators showing if key requirements are met.
    - **`aggregated_scores`**: Overall scores for macro and micro criteria.
  - **`personal_info`**: Extracted personal details (e.g., name, contact details, current position).
  - **`valid_resume_and_jd`**: Boolean indicating if both resume and JD are valid for evaluation.

- **`details`**:
  - **Resume Analysis**: Detailed breakdown of education, certifications, skills, project history, and professional experience.

### Dataset Preparation Methodology
1. **JD Generation**: Resumes were randomly sampled, and GPT-4o generated job descriptions tailored to these resumes.
2. **JD Comparison**: Individual resumes were then compared to a randomly generated JD using GPT-4o to produce relevance scores and justifications.

### Example Entry
A sample JSON object in this dataset resembles the following structure:

```json
{
  "input": {
    "job_description": "Full job description text...",
    "macro_dict": {"experience": 89, "strategic thinking": 11},
    "micro_dict": {"market research": 7, "it and manufacturing sector knowledge": 93},
    "additional_info": "Preferred candidates are from top-tier institutes...",
    "minimum_requirements": ["5+ years of experience...", "Strong understanding of IT..."],
    "resume": "Resume text with skills, experience, etc."
  },
  "output": {
    "justification": ["Candidate has only 1.5 years of experience, below the required 5+ years..."],
    "scores": {
      "macro_scores": [{"criteria": "experience", "score": 3}, {"criteria": "strategic thinking", "score": 2}],
      "micro_scores": [{"criteria": "market research", "score": 4}, {"criteria": "it and manufacturing sector knowledge", "score": 3}],
      "requirements": [{"criteria": "5+ years of experience...", "meets": false}, ...],
      "aggregated_scores": {"macro_scores": 2.89, "micro_scores": 3.07}
    },
    "personal_info": {"name": "Muhammad Talha Riaz", "email": "talhariaz9969@gmail.com", ...},
    "valid_resume_and_jd": true
  },
  "details": {
    "name": "Talha Riaz",
    "skills": ["HTML", "CSS", "JavaScript", ...],
    "education": [{"university": "University of the Punjab", "degree_title": "BS Management", "end_date": "06-2021"}],
    ...
  }
}
```

### Use Cases
This dataset is designed to support research in:
- **AI-driven recruitment**: Assessing resume-JD alignment and scoring accuracy.
- **Job Matching Algorithms**: Testing algorithms that rank or filter candidates based on job fit.
- **Natural Language Processing (NLP)**: Analyzing how NLP can evaluate resume relevance based on custom criteria.

### Licensing and Citation
Please cite this dataset as follows:
```plaintext
Dataset generated using GPT-4o by [rohan/netsol].
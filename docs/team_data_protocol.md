# Team Data Collection Protocol

## Goal

Ensure each group member independently contributes 10,000 comments and that contributions are auditable.

## Required Per Member

- A dedicated scraper run command and output file.
- One CSV file with around 10,000 comments.
- A short note in commit message about date, mode, and source scope.

## Naming Convention

- `member_a_10k.csv`
- `member_b_10k.csv`
- `member_c_10k.csv` (if applicable)

Store all files in:

- `data/raw/member_submissions/`

## Evidence of Contribution

Each member should:

1. Run scraper from their own branch.
2. Commit script/config updates (not large CSV files).
3. Share screenshots of terminal output (rows collected).
4. Open a pull request with a short summary.

## Recommended Tracking Table (for report appendix)

- Member name
- Date collected
- Video scope used
- Raw rows collected
- Rows after dedup
- Final accepted rows


# Input preprocessing (label-agnostic; applied once before splitting)

Implemented in `src/common.py` (`PREPROC`). The prescriptions are never used.

| Step | Inputs affected |
|---|---|
| Constant in all records (removed) | midfoot amputation; other amputation |
| Exact duplicate of another input (removed; first-named kept) | sensory neuropathy (= neuropathy present); foot structure/shape (= deformity present); shopping (= walking/stairs); needs specialised transport (= access to clinic); previous footwear used (encoded by previous-footwear outcome = not applicable) |
| Structural missingness | deformity severity recorded only when a deformity is present (125 records) and coded "not applicable" |
| "Not documented" | kept as an explicit category (most often diabetes type, 70 records; gender, 19) |
| Rare levels (< 10 records) merged into the clinically nearest level | lesion type: corn-containing types (1–3 records each) merged into their callus counterparts, giving ulcer only / ulcer with hyperkeratosis / hyperkeratosis only / none; specific site: IPJ hallux (8) into hallux; employment: unemployed (6) and retired as not employed |
| Kept although rare | binary inputs such as equinus (9 records), footwear aesthetics important (9), gardening (6) |

Result: 58 inputs (55 categorical, 3 numeric). After these steps no input is constant in any training fold, no two records share identical inputs, and every class of every decision occurs at least 15 times in each training fold and 3 times in each test fold. A run without these steps (`PAKDD_PREPROC=0`) reached the same conclusions.

# Dataset audit: DravidianCodeMix Tamil offensive (6 classes)

## Label counts

| label | train raw | dev raw | test raw | train clean |
|---|---|---|---|---|
| Not_offensive | 25425 | 3193 | 3190 | 25065 |
| Offensive_Untargetede | 2906 | 356 | 368 | 2883 |
| Offensive_Targeted_Insult_Individual | 2343 | 307 | 315 | 2332 |
| Offensive_Targeted_Insult_Group | 2557 | 295 | 288 | 2541 |
| Offensive_Targeted_Insult_Other | 454 | 65 | 71 | 450 |
| not-Tamil | 1454 | 172 | 160 | 1441 |
| **total** | 35139 | 4388 | 4392 | 34712 |

## Leakage and duplicates

- duplicate texts inside train: 298
- test texts also in train: 77
- dev texts also in train: 69
- train rows removed by cleaning: 427 (duplicates collapsed to the majority label, overlap with dev/test dropped)

## Text properties (train)

- script share: roman 81.6%, tamil 18.3%, other 0.2%
- words per comment: median 8, p95 22, p99 43
- comments with @mentions: 117
- comments with URLs: 0


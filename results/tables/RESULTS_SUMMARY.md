# Results summary

## AUC trajectory (frozen R0 classifier, held-out 30% humans)

| round   |   STYLE7 |   STYLO |
|:--------|---------:|--------:|
| R0      |    0.999 |   0.991 |
| R1      |    0.629 |   0.926 |
| R2      |    0.079 |   0.776 |

## LOF — % humans outside the LLM STYLO cloud

|    | outside   |
|:---|:----------|
| R0 | 100%      |
| R1 | 83%       |
| R2 | 59%       |

## STYLE7 feature means (Human vs rounds)

|                    |   Human |      R0 |     R1 |     R2 |
|:-------------------|--------:|--------:|-------:|-------:|
| n_words            |  41.34  | 244.266 | 71.121 | 38.007 |
| avg_word_len       |   5.142 |   5.959 |  4.923 |  4.364 |
| avg_sent_len       |  17.433 |  43.79  | 25.647 | 14.215 |
| lexical_diversity  |   0.827 |   0.696 |  0.824 |  0.875 |
| first_person_ratio |   0.032 |   0.008 |  0.047 |  0.097 |
| hedging_ratio      |   0.004 |   0     |  0.008 |  0.021 |
| comma_density      |   0.057 |   0.087 |  0.054 |  0.003 |

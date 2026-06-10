# LLM과 학생 답변의 차이와 특징 분석

**COSE471 Data Science · Term Project (Korea University, 2026)**

> Research Question: *학생 답변과 LLM 답변을 구별할 수 있는가? 구별된다면 무엇이 'LLM 티'이고,
> 프롬프트로 그 차이를 어디까지 지울 수 있으며, 끝까지 안 지워지는 '인간 고유 서명'은 무엇인가?*

수업 필수 기법 중 **Classification** 과 **Outlier Detection** 2가지를 핵심으로 사용한다
(+ 필수 항목인 data understanding/preprocessing).

---

## 1. 데이터 요약

| 단계 | 위치 | 내용 |
|------|------|------|
| 원본 | `data/raw/26S_COSE471_Discussion_Datasets.xlsx` | 5개 Discussion 시트(=5 topics), 9개 질문 컬럼, 294 student-rows, ~155k 글자 |
| 전처리(master) | `data/preprocessed/Q1~Q9_corrected.json` | 질문별 분리, 응답당 14개 파생 필드(JSON object), 총 542 응답 |
| 배치(재현용) | `data/preprocessed/{input_json_batches, claude_prompt_ready_batches}` | 전처리 입력/프롬프트-주입 배치 |

**원본 → 전역 질문 매핑**

| 시트 | 컬럼 B | 컬럼 C | stage |
|------|--------|--------|-------|
| Discussion 1 | Q1 | Q2 | `pre` (수강동기·활용분야) |
| Discussion 2 | Q3 | Q4 | `discussion_1` (Data/preprocessing) |
| Discussion 3 | Q5 | Q6 | `discussion_2` (Pattern mining) |
| Discussion 4 | Q7 | Q8 | `discussion_3` (Clustering) |
| Discussion 5 | Q9 | — | `discussion_4` (Outlier detection) |

**master JSON 필드**: `id, original_no, question, stage, raw_text, language,
cleaned_text, english_text, one_sentence_summary, keywords, themes, quality,
reason_for_quality, confidence`
(원문 보존 + 분석용 파생 필드. cleaned_text=다국어판 / english_text=영어판 main text 선택용)

질문별 응답 수(전처리 후): Q1=52, Q2=52, Q3=66, Q4=66, Q5=63, Q6=63, Q7=67, Q8=67, Q9=46.
저품질(`quality:low`)은 **삭제하지 않고 플래그만** 했다(질문별 low: 0~24개).

---

## 2. 파이프라인 (최종 설계)

**대전제: 내용은 키워드로 R0~R2 전부 고정 → 인간 vs LLM 차이 = 오직 말투, 라운드 간 차이 = 오직 말투 코칭.**

```
[1] 전처리          raw xlsx ──(고정 프롬프트, Claude)──▶ Q*_corrected.json   (완료)
[2] R0~R2 생성      키워드(내용고정) + 말투지시만 변화 ──(gpt-5-mini·ELICE)──▶ R0/R1/R2.json (각 1:1, 421개)
                   R0=naive · R1=가벼운 코칭 · R2=공격적 회피   (슬라이드·임베딩 미사용)
[3] R0 분석        질문별 Classifier(LogReg+GBM) AUC on STYLE7 (+길이 baseline·bootstrap CI·permutation)
                   독립 판정자 STYLO 교차 / 음성통제(인간-인간·라벨셔플 AUC≈0.5)
[5] 라운드 평가     R0 분류기 freeze + 인간 30% 잠금 → R0/R1/R2 AUC 궤적
                   STYLE7(겨냥, 뚫림) vs STYLO(비겨냥, 유지) → 표면 vs 근본 분해
[기법②] Outlier   STYLO 공간 LOF → LLM이 끝까지 못 덮는 '인간 고유 서명' 검출
```

핵심 공정성: persona·temperature·top_p·**키워드**를 전 라운드 고정, 오직 '말투 지시'만 변경
→ 모든 ΔAUC = **순수 말투 효과** (R0→R1 가벼운 코칭, R1→R2 공격적 코칭).
내용 분석(임베딩 유사도, 옛 3-A)은 최종안에서 **제외**(참여도·임계 모호) — 내용은 '측정'이 아니라 '고정'.

---

## 3. 폴더 구조

```
DS_project/
├── README.md                  # 이 문서
├── requirements.txt
├── config/config.yaml         # 경로·시드·모델·기법 파라미터 (단일 진실원천)
├── prompts/                   # 전처리/생성 프롬프트 (재현성)
│   ├── preprocessing_prompt.txt
│   └── generation_R0/R1/R2.md
├── data/
│   ├── raw/                   # 원본 xlsx (제공됨)
│   ├── preprocessed/          # Q*_corrected.json (제공됨) + 배치
│   ├── external/slides/       # 강의 슬라이드 (grounding 입력 — 추가 필요)
│   ├── llm_answers/           # R0.json, R1.json, R2.json (생성 산출물)
│   └── features/              # style7/stylo/embedding 행렬 캐시
├── src/
│   ├── data_io.py             # master/round 로드, 질문별 라벨 테이블(인간0/LLM1)
│   ├── preprocessing/         # xlsx→master 빌드(대부분 완료, API 재현용)
│   ├── generation/            # R0/R1/R2 생성 (외부 LLM API)
│   ├── features/              # style7.py / stylo.py / embeddings.py
│   ├── analysis/              # content_similarity / classify / outlier_lof
│   └── evaluation/            # round_eval (freeze+holdout AUC 궤적)
├── notebooks/                 # 탐색 + 그림 생성
├── results/{figures,tables}/  # 산출 그림·표
└── report/                    # 최종 LaTeX 보고서 (≤10p)
```

---

## 4. 재현 방법

```bash
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -r requirements.txt
# 파라미터는 모두 config/config.yaml 에서 조정
```
실행 순서(예정): `generation(R0~R2) → features(style7/stylo) → classify(R0) → evaluation(step5) → outlier(LOF) → 그림`

---

## 5. 현재 상태 / 남은 결정

- ✅ **생성 API**: ELICE `gpt-5-mini` 연결 완료 (`config/secrets.yaml`, git 제외)
- ✅ **GPU 불필요**: 내용분석·임베딩 제외 → 전 과정 CPU
- ✅ **슬라이드**: 생성에 미사용(키워드가 내용 제공). 추출 텍스트는 참고용 보관.
- ✅ **분석 main text**: `english_text` 확정
- ❔ **매칭 인간 N**: 403(non-low, 권장) vs 510(저품질 포함). 플랜의 "426"은 이전 버전 수치. (`human_set`)
- ❔ **생성 프롬프트**: 키워드만 vs 질문 framing 약간 추가
- ⚠️ **표본**: 질문당 ~80개(인간≈45 + LLM 1:1) → bootstrap CI 넓음(인지하고 진행)
- ℹ️ **ESL 교란**: Limitation & Future Work로 처리(최종안 5-f) — 별도 생성 라운드 불필요
```

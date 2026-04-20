# Figure Improvement Status — 2026-04-20

## 목적
`sample-sigconf.pdf`의 figure 가독성 개선.
ACM sigconf 2-column format에서 figure들이 너무 작거나(단일 컬럼에 배치), 텍스트가 안 보이는 문제 해결.

---

## 완료된 작업

### 1. `.tex` 레이아웃 변경
- **Q2 (Fig 3), Q3 (Fig 4), Q5 (Fig 5)**: `figure` → `figure*` (전체 텍스트 너비)
- minipage 너비: `0.48\columnwidth` → `0.48\textwidth`, `0.31\columnwidth` → `0.31\textwidth`
- **부록 Fig 6–11** 전체: `figure` → `figure*`
- Fig 6 (special bins): `width=0.92\textwidth`
- Fig 9 (behavior slices): `0.58\textwidth` + `0.38\textwidth`
- Fig 11 (transfer): `0.92\textwidth`

### 2. Figure 생성 노트북 수정 (figsize/폰트 확대)

| 노트북 | 수정 내용 |
|--------|----------|
| `02_q2_routing_control.ipynb` | figsize `4.3×3.8` → `5.2×4.2` |
| `03_q3_design_justification.ipynb` | figsize `6.0×3.9` → `6.5×4.4`, x-tick rotation 제거, 레이블 줄바꿈 추가 |
| `05_q5_behavior_semantics.ipynb` | heatmap figsize 확대, `annot_kws={'size': 11}`, 축 레이블 fontsize=11 |
| `appendix/A02_appendix_structural_ablation.ipynb` | figsize `5.5×3.5` → `6.5×4.2`, 절대경로로 save path 수정 |
| `appendix/A03_appendix_sparse_and_diagnostics.ipynb` | figsize 전체 확대, 절대경로 수정, combined save 추가 |
| `appendix/A04_appendix_behavior_and_bins.ipynb` | figsize 확대, combined 2-panel save 추가, 절대경로 수정 |
| `appendix/A05_appendix_interventions_and_cases.ipynb` | figsize 확대, save 추가, 절대경로 수정 |
| `appendix/A06_appendix_optional_transfer.ipynb` | combined 2-panel save 추가, 절대경로 수정 |

### 3. 데이터 CSV 수정 (노트북이 읽는 그룹/레이블명 불일치 해결)

| 파일 | 수정 내용 |
|------|----------|
| `appendix/data/appendix_special_bins.csv` | `group` 값: `short` → `short (1-3)`, `medium (4-8)`, `long (9+)`, `tail (1-5)`, `mid (6-20)`, `head (21+)` |
| `appendix/data/appendix_structural_variants.csv` | `variant_group` 리맵핑 (`family_prior`→`cue_org` 등), 레이블 수정, `Groups Shuffled` 항목 추가 |
| `appendix/data/appendix_sparse_diagnostics.csv` | `'Top-3gr Top-2ex (6 act.)'` → `'Top-3gr Top-2ex — main'` |
| `appendix/data/appendix_routing_diagnostics.csv` | 동일 레이블 수정 |
| `appendix/data/appendix_sparse_tradeoff.csv` | 동일 레이블 수정, `active_experts` 컬럼 추가 |

### 4. 현재 PDF 상태
- **21페이지**, 0 overfull 에러, 1,017,263 bytes
- 모든 figure에 실제 데이터 표시됨
- Fig 3 (Q2), Fig 4 (Q3): full text-width 2-panel
- Fig 5 (Q5): full text-width 3-panel heatmap
- Fig 6–11 (부록): 모두 full text-width

---

## 남은 개선 사항 (향후 작업)

### 우선순위 높음
1. **노트북 재실행 확인 필요**: 일부 노트북(A03–A06)이 session에서 재실행되었는지 확인 필요. 현재 figures/appendix/ 폴더에 파일이 있지만, 최신 코드 변경 후 재실행된 것인지 불확실.
   - 확인 방법: `ls -la figures/appendix/` 타임스탬프 확인

2. **Fig 9(b) relative gain bar**: y-axis 스케일이 너무 넓어 보임 — A04 노트북에서 ylim 조정 고려

3. **Fig 7(b) image (stage-semantics)**: 서브피규어 크기가 여전히 작음 — A02 노트북에서 (b) 패널 추가 확대 필요

### 우선순위 보통
4. **Fig 8 heatmap 레이블**: `memory_plus`, `focus_plus` 등 x-tick 레이블이 인쇄 크기에서 작을 수 있음
5. **Fig 10(a) x-axis**: 개입 레이블(Repeat-heavy 등) 줄바꿈/rotation 확인
6. **Fig 6 y-axis range**: 0.0–0.40 범위가 너무 넓음, y-axis를 실제 데이터 범위에 맞게 축소

### 낮은 우선순위
7. **Q5 (Fig 5) combined heatmap**: 현재 3 datasets × 3 cases = 9 패널인데, 레이아웃 최적화 여지 있음
8. **전체 텍스트 폰트 consistency**: 모든 figure의 axis label/tick 폰트가 통일되어 있는지 전체 확인

---

## 파일 위치 참조

```
writing/ACM_template/
  sample-sigconf.tex          # 메인 LaTeX 파일
  sample-sigconf.pdf          # 빌드된 PDF
  figures/
    fig_q2_routing_control_a.pdf
    fig_q2_routing_control_b.pdf
    fig_q3_design_justification_a.pdf
    fig_q3_design_justification_b.pdf
    appendix/
      a02_structural_cue_org.pdf
      a02_structural_temporal.pdf
      a03_objective_variants.pdf
      a03_routing_diagnostics_lines.pdf
      a03_routing_heatmaps.pdf
      a03_sparse_variants.pdf
      a04_behavior_slices.pdf
      a04_slice_relative_gain.pdf
      a04_special_bins.pdf
      a05_intervention_score_drop.pdf
      a05_routing_profiles.pdf  (a/b/c 개별 파일도 존재)
      a06_transfer.pdf

writing/260419_real_final_exp/
  02_q2_routing_control.ipynb
  03_q3_design_justification.ipynb
  05_q5_behavior_semantics.ipynb
  appendix/
    A02_appendix_structural_ablation.ipynb
    A03_appendix_sparse_and_diagnostics.ipynb
    A04_appendix_behavior_and_bins.ipynb
    A05_appendix_interventions_and_cases.ipynb
    A06_appendix_optional_transfer.ipynb
    data/   # CSVs (모두 수정됨)
```

---

## PDF 재빌드 방법

```bash
cd /workspace/FeaturedMoE/writing/ACM_template
pdflatex -interaction=nonstopmode sample-sigconf.tex
```

노트북 재실행 (figure 재생성):
```bash
cd /workspace/FeaturedMoE/writing/260419_real_final_exp
jupyter nbconvert --to notebook --execute --inplace 02_q2_routing_control.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_q3_design_justification.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_q5_behavior_semantics.ipynb
cd appendix
jupyter nbconvert --to notebook --execute --inplace A02_appendix_structural_ablation.ipynb
jupyter nbconvert --to notebook --execute --inplace A03_appendix_sparse_and_diagnostics.ipynb
jupyter nbconvert --to notebook --execute --inplace A04_appendix_behavior_and_bins.ipynb
jupyter nbconvert --to notebook --execute --inplace A05_appendix_interventions_and_cases.ipynb
jupyter nbconvert --to notebook --execute --inplace A06_appendix_optional_transfer.ipynb
```

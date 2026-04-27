# Position Notebook Index

이 폴더는 paper의 figure/table 위치별 notebook 초안과 공통 스타일 helper를 담는다.

현재 버전의 목적은 단순 plotting demo가 아니라, 각 slot이 어떤 claim을 증명해야 하는지와 어떤 export / logging이 필요한지를 notebook 자체에 내장하는 것이다.

## 핵심 파일

- `slot_viz_helpers.py`: 공통 스타일, dual-metric bar+line plot, heatmap, table styler helper
- `generate_position_assets.py`: sample CSV와 위치별 ipynb 생성 스크립트
- `data/`: notebook이 읽는 값 템플릿 CSV
- `q2_q5_main_body_strategy.md`: 본문 Q2-Q5 재설계 메모

## 생성되는 notebook 목록

- `01_main_results_table.ipynb`: main overall table 위치
- `02_q2_routing_control.ipynb`: Q2 routing control figure
- `03_q3_stage_structure.ipynb`: Q3 stage structure figure
- `04_q4_lightweight_cues.ipynb`: Q4 lightweight cue figure
- `05_q5_behavior_regimes.ipynb`: Q5 feature intervention figure
- `A01_appendix_main_configuration.ipynb`: fixed main configuration table
- `A02_appendix_implementation_spec.ipynb`: implementation spec table
- `A03_appendix_feature_families.ipynb`: representative cue family table
- `A04_appendix_datasets.ipynb`: dataset table
- `A05_appendix_full_results.ipynb`: full results table
- `A06_appendix_extended_structure.ipynb`: extended structural ablation figure
- `A07_appendix_routing_diagnostics.ipynb`: routing diagnostics figure
- `A08_appendix_behavior_slices.ipynb`: appendix behavior-slice figure
- `A09_appendix_transfer_variants.ipynb`: transfer or portability figure

## 사용 방식

1. `data/` 아래 CSV 값을 수정한다.
2. 해당 위치의 notebook만 열어서 실행한다.
3. 저장은 notebook 내부에서 하지 않는다. 지금 버전은 화면 preview 전용이다.

## 이번 버전에서 달라진 점

- Q2 main slot은 dataset별 quality small multiples + stage-averaged group profile + compact case heatmaps 구조로 바뀌었다.
- Q3 main slot은 stage-removal을 본문에서 빼고, final 3-stage vs reduced-stage alternatives + slow-to-fast order neighborhood만 남긴다.
- Q4 panel (b)는 family-profile retention 대신 metadata 제거 후에도 routing spread와 active experts가 유지되는지 보여준다.
- Q4 notebook은 `variant_or_model` 기반 CSV를 읽어도 자동으로 `cue_setting`으로 정규화한다.
- Q5 main slot은 semantic intervention quality + regime-selective routing gain scatter로 바뀌었다.
- Q2-Q5 notebook은 실행 시 지금 값이 draft-fill인지, 아니면 추가 logging이 필요한 placeholder인지 바로 표시한다.
- 추가 logging이 필요한 panel은 notebook 출력에 필요한 export column과 recommended logging key를 같이 적어둔다.

## data/ CSV 해석

- `draft_fill`: 지금 당장 디자인 확인용으로 써도 되는 1차 값
- `placeholder_requires_logging`: 구조만 잡아 둔 상태이며, 실제 figure를 채우려면 추가 diag export가 필요함

## Figure 텍스트 규칙

- plot title, axis label, legend, tick label 안에는 한글을 넣지 않는다.
- figure 안의 문장은 최대한 짧게 유지한다.
- 상세 설명은 notebook markdown / print output에서 처리한다.

## 디자인 원칙

- metric comparison류는 `NDCG@20` 막대 + `HR@10` 선 그래프를 기본값으로 사용
- bar와 line은 같은 variant 색을 공유하고, bar는 반투명, line은 진한 색과 marker로 표시
- y-axis는 zero-base를 강제하지 않고 tight range를 써서 차이가 잘 보이게 함
- table은 pandas Styler로 paper-like header, 얇은 grid, best/second 강조를 적용
- 본문 figure는 가능한 한 direct evidence 위주로 두고, indirect diagnostic은 appendix notebook으로 내린다.
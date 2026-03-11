# HGRv3 Structure

## Goal
- HGR의 outer rule distill을 버리고, inner clone routing에만 rule teacher를 건다.
- outer는 feature를 쓰지 않고 hidden-only로 단순화한다.

## Flow
1. global / stage transformer는 HGR와 동일한 scaffold를 유지한다.
2. 각 stage outer router는 hidden context만 보고 4개 group weight를 만든다.
3. 각 group 내부 learned router는 `hidden + group feature` interaction으로 clone logits를 만든다.
4. group-local inner bin teacher는 group feature ratio 평균으로 clone teacher logits를 만든다.
5. inner rule 모드:
   - `off`: learned clone logits만 사용
   - `distill`: learned clone logits만 사용, teacher KL만 추가
   - `fused_bias`: `learned + bias_scale * teacher`로 inference logits 형성
   - `distill_and_fused_bias`: distill + fused bias 둘 다 적용
6. 최종 gate는 `group_weight * clone_weight`다.

## Main Differences vs HGR
- outer router:
  - HGR: hidden + stage/group feature
  - HGRv3: hidden-only
- teacher 위치:
  - HGR: outer group logits distill
  - HGRv3: inner clone logits distill / fusion
- rule semantics:
  - HGR: stage-level 4-group guidance
  - HGRv3: group-local bin guidance

## Inner Teacher
- input: 해당 group raw feature subset
- normalize: ratio space
- score: feature mean ratio
- bin centers: 기본 `expert_scale=4`, 균등 배치
- logits: `-sharpness * (score - center)^2`

## R0 Quick Run
- fixed layout: `15 = [1,0,2,0,1,0,0,0]`
- merge: `serial`
- anchors:
  - `A0 = 128 / 16 / 160 / 64`
  - `A1 = 160 / 16 / 256 / 112`
- default clone count per group: `4`
- modes:
  - `off`
  - weak `distill` only in `R0`
- `R0` compares `expert_top_k = 1 / 2 / 4`
- strong distill / fused bias are deferred to `R1`

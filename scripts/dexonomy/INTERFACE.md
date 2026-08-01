# Dexonomy 접촉 테이블 인터페이스 명세 (접촉 추출 세션 → GT 생성 세션)

버전 2 (2026-07-23). GT 계획서 D8 항목("접촉 추출 인터페이스 합의")에 대한 접촉 추출 세션의 공식 명세.
문의처: `scripts/dexonomy/` (추출 코드), `data/dexonomy/contact_cache/` (산출물).

## 1. grasp_id 규약

파지 하나의 전역 식별자는 **(succ_collect 기준 상대경로, 파일 내 행 번호)** 쌍이며,
문자열 직렬화 형식은 다음과 같다:

```
<타입>/<물체ID>/floating/<scaleNNN>.npy#<i>
예: 24_Tip_Pinch/3f1156b006cc435fa840c6055a775901/floating/scale008.npy#3
```

- `i`는 **0-기반** 행 번호. 원본 npy의 `grasp_qpos[i]` 행과 접촉 npz의 모든 배열 `[i]` 행이 같은 파지다.
- 경로 구분자는 `/`, id-행 구분자는 `#`. 확장자는 원본 기준 `.npy`로 표기한다.
- **행 순서 보존 보장**: 접촉 테이블은 원본 npy의 grasp 순서를 절대 재정렬하지 않는다.
  파생 테이블(GT parquet 등)도 이 규약을 따르고, 재정렬이 필요하면 grasp_id 열을 반드시 포함할 것.
- join 방법: 접촉 npz 경로 = `contact_cache/` + (상대경로의 `.npy` → `.npz`).
  원본 qpos(관절각 22개 포함) = `succ_collect/` + 상대경로의 `grasp_qpos[i]`.

## 2. 데이터 규모 (전수 실측, 추정 아님)

- 파지 총수 **9,475,172** / 파일 **1,501,599** (파일당 평균 6.3개, 최소 1, 최대 ~27).
  ※ GT 계획서의 "파일당 grasp 1개"는 사실과 다름 — 규모 추정 시 파지 수 기준으로 계산할 것.
- 파일 = (타입 31 × 물체 10,691 × 스케일 구간 6) 조합 중 성공 파지가 있는 75.5%.
- 물체 mesh는 물체ID당 1개(`processed_data/<물체ID>/mesh/simplified.obj`, 단위 정규화)이며
  파일·파지 간 변형 없음. 스케일만 다름.

## 3. 스케일 규약 (중요)

- 파지 i의 실제 물체 크기 배율: `total_scale[i] = (scaleNNN의 NNN/100) × scene_scale[i]` (npz에 저장됨).
- 단위 mesh 좌표 × total_scale = 파지 좌표계(= 접촉점·qpos가 표현된 world 좌표계). 물체는 원점, 회전 없음.
- **scaleNNN 폴더명 ≠ 실제 크기.** 실측 결과 6개 구간 모두 실제 크기 분포가 사실상 동일
  (5~95 percentile ≈ 5.2~19.4 cm, 중앙값 ≈ 10 cm). scene_scale이 구간 차이를 상쇄하도록 만들어져 있다.
- 따라서 **한 구간만 사용 = 전체를 균일하게 1/6 다운샘플** (~158만 파지)과 동치. 타입·물체·크기 커버리지 유지됨.

## 4. 접촉 npz 스키마 (버전 2)

경로: `contact_cache/<타입>/<물체ID>/floating/<scaleNNN>.npz` (원본과 1:1). 링크 L=23, 손가락 유닛 F=6.
링크 이름·순서, 유닛 이름·순서(`["index","middle","ring","little","thumb","palm"]`),
링크→유닛 매핑(`link_finger_id`)은 `contact_cache/meta.json`에 명시.

| 필드 | shape | 의미 |
|---|---|---|
| `link_min_dist` | (N,L) f32 | 링크 표면 샘플점과 물체 표면 사이 signed 최소거리 (m, 음수=침투) |
| `link_contact` | (N,L) bool | `link_min_dist < threshold` (기본 3mm — §5 참조) |
| `link_contact_point` | (N,L,3) f32 | 최소거리 지점의 물체 표면 최근접점 (파지 좌표계, m) |
| `link_contact_normal` | (N,L,3) f32 | 그 지점의 물체 표면 법선 (물체 바깥 방향) |
| `link_contact_area` | (N,L) f32 | 임계 내 샘플점의 대표면적 합 = 링크 접촉 면적 근사 (m²) |
| `link_n_contact_pts` | (N,L) i16 | 임계 내 샘플점 개수 (면적 신뢰도 판단용) |
| `link_patch_centroid` | (N,L,3) f32 | 임계 내 접촉점들의 면적 가중 centroid. 비접촉 시 최근접점으로 폴백(area=0으로 구분) |
| `finger_*` | (N,F,·) | 위 필드들의 유닛(손가락/손바닥) 단위 병합: min_dist는 소속 링크 min, area는 합, centroid는 면적 가중 평균 |
| `finger_contact_link` | (N,F) i8 | 유닛 내 최근접 링크의 전역 링크 index |
| `total_scale` | (N,) f32 | §3의 실제 크기 배율 |

- D2의 유닛 병합(같은 손가락 링크 병합 + 손바닥 독립)은 `finger_*` 필드가 이미 수행한 것과 동일.
  탐색 중심으로는 `finger_patch_centroid`를 표면에 투영해 사용 권장 (D2 명세 그대로).
- D8의 "접촉 면적"은 `finger_contact_area` / `link_contact_area`.
- 곡률은 이 테이블에 없음 — GT 계획 P2(물체 전처리)에서 물체당 1회 계산 후 접촉점 최근접 join 권장.

## 5. threshold 합의 사항 (D8의 "2mm" 재검토 필요)

D8은 Dexonomy 기준(2mm)을 지정했으나, **측정 방식이 달라 같은 2mm가 아니다**:

| | Dexonomy 원본 | 본 접촉 테이블 |
|---|---|---|
| 거리 정의 | MuJoCo margin, geom 간 정확 거리 | 손 표면 샘플점(896개/파지) ↔ 물체 SDF |
| 물체 형상 | coacd convex 조각 | simplified mesh |
| 오차 성질 | 정확 | 거리 과대추정 (샘플 이산화; 독립 표본 간 min-dist 차이 평균 ~1mm) |

실측 근거: 2mm를 그대로 쓰면 실제로 닿은 손가락(측정값 2.0~3.8mm)이 타입별로 수 % 누락된다
(Lateral/Ring/Writing_Tripod 등에서 확인). **3mm 사용을 권장**하며, taxonomy별 접촉 패턴이
정의와 일치함을 확인했다 (`contact_cache/summary.json`).

임계값 스윕 실측 (v2 테이블, 표본 190,122 파지):

| threshold | 손가락 ≥2개 접촉 | 평균 접촉 손가락 수 | palm 접촉율 | 접촉 0개 |
|---|---|---|---|---|
| 1.5mm | 92.2% | 3.24 | 13.6% | 0.77% |
| 2.0mm | 97.8% | 3.67 | 20.8% | 0.07% |
| 2.5mm | 99.3% | 3.86 | 26.4% | 0.01% |
| **3.0mm (권장)** | 99.8% | 3.94 | 30.4% | 0.00% |
| 4.0mm | 100.0% | 4.03 | 35.8% | 0.00% |

트레이드오프: 임계값을 올리면 누락은 줄지만 경계상의 palm 접촉이 늘어난다(2mm 21% → 3mm 30%).
GT 세션이 palm 유닛의 앵커 참여에 민감하다면 `finger_min_dist[:,5]`로 palm만 다른 임계값을
적용하는 것도 가능하다 (raw 거리 저장 덕분에 자유).

- flag(`*_contact`)는 3mm로 저장되어 있으나 **raw 거리(`*_min_dist`)가 있으므로 GT 세션이
  임의 임계값으로 재유도 가능** (`dist < 0.002` 한 줄). 재계산 불필요.
- 단, `*_contact_area` / `*_patch_centroid`는 저장 시점 임계값(3mm)으로 계산됨.
  다른 임계값의 면적·centroid가 필요하면 재실행 요청 (전체 ~30분).

## 6. 알려진 한계

1. 거리 과대추정: 샘플 이산화로 링크당 ~1mm 수준 노이즈 (distal은 64점으로 더 정밀, knuckle 등 16점 링크는 더 큼).
2. non-watertight mesh 이상치: inside/outside 판정 실패로 큰 음수 거리(최대 -111mm)가 링크 기준 0.0005% 존재.
   `link_min_dist < -0.01` 필터 권장.
3. squeeze_qpos 미사용 (D8 합의대로 grasp_qpos 기준). 필요 시 `--qpos-key`로 별도 테이블 생성 가능.

## 7. 재현 방법

```
python scripts/dexonomy/setup_assets.py --inspect   # asset 점검
python scripts/dexonomy/validate_fk.py --parity     # FK 검증 (MuJoCo 대비 1.6e-8 m)
python scripts/dexonomy/extract_contacts.py         # 추출 (resume 지원)
python scripts/dexonomy/summarize_contacts.py       # 통계
```
파라미터 버전: `contact_cache/meta.json`의 `version` (현재 2 = 면적·centroid 포함).

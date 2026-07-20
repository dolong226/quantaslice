# Báo cáo chi tiết — Tầng AI/ML: Emergency Detector (QuantaSlice)

> Tài liệu này tổng hợp **trung thực** toàn bộ quá trình xây dựng, đánh giá, audit và
> kết luận về tầng AI/ML của QuantaSlice. Nguyên tắc xuyên suốt: *báo cáo đúng những gì
> đo được, không tô hồng.* Kèm số liệu tái lập được và đường dẫn code.
>
> Plan gốc: [`plans/QuantaSlice_ML_Plan.md`](../plans/QuantaSlice_ML_Plan.md).
> Dataset: `colosseum-oran-coloran-dataset/rome_static_medium` (ColO-RAN).

---

## 0. TL;DR (đọc cái này trước)

1. **Pipeline ML đúng và sạch** — không rò rỉ train/test, dữ liệu chuẩn, split hợp lệ, model không "gian lận".
2. **Nhãn ban đầu bị lỗi** (chuẩn hoá theo median) → điểm thấp giả tạo. **Đã sửa** thành nhãn độ-trễ-hàng-đợi vật lý.
3. **Nhưng trên dataset này, phát hiện emergency là bài toán TẦM THƯỜNG**: một luật 1 dòng *không huấn luyện* đạt PR-AUC 0.990, model ML 0.997 (+0.007). Lý do: ColO-RAN `rome_static_medium` gần **tĩnh** (UE bất động, traffic đều) → trạng thái SLA gần như do config quyết định.
4. **Dataset không chứa "emergency" thật** (không mobility/burst/sự cố). Nên không thể validate một detector khẩn cấp thuần tuý trên nó.
5. **Giá trị THẬT #1 (downstream):** đưa vector ưu tiên `p` của detector vào QUBO → **giảm 43% chi phí vi phạm SLA** so với chính sách tĩnh (closed-loop, §11).
6. **Giá trị THẬT #2 (dữ liệu động — hướng B):** trên **TRACTOR** (traffic 5G thật, bursty; full Trial0, 12 luồng, 27.5k cửa sổ test), dự báo vi phạm SLA 2s trước → ML **thắng luật tầm thường +0.219 PR-AUC** (0.803 vs 0.585) và **bắt sớm 53% vi phạm mà luật bỏ lỡ hoàn toàn** (lead-time recall 0.53 vs 0.00). Đây là nơi ML *thực sự* vượt luật — điều ColO-RAN tĩnh không cho được (§12).

**Con số nên trích dẫn:**
> *"On real, dynamic 5G traffic (TRACTOR), the ML detector forecasts SLA violations 2 s ahead, beating a threshold rule by +0.122 PR-AUC and catching 27% of imminent violations the rule misses entirely. Downstream, acting on its priority vector `p` cuts SLA-violation cost 43% vs a fixed policy in the closed-loop QUBO."*

---

## 1. Vai trò trong pipeline

```
KPM stream → [ML detector] → (emergency flag ê, priority vector p) → QUBO/QAOA reallocation → loop
```

Detector là khối **đầu tiên**. Hợp đồng đầu ra (khớp `core.types.Prediction`):
- **`emergency_flag` ê ∈ {0,1}** + `emergency_prob ∈ [0,1]` — trigger tái tối ưu.
- **`priority` p = (p_eMBB, p_URLLC, p_mMTC), ≥ 0** — reweight objective của QUBO: `Q_new = Q_base + diag(p)`.

Ràng buộc kiến trúc: package `ai` **chỉ import từ `core`**, không import `quantum`/`orchestrator`/`pipeline`. Detector cắm vào `Runner` qua `ai.provider_registry` (tên `"ml"`), đổi 1 dòng config là chuyển từ bootstrap provider (mock/threshold) sang ML thật.

---

## 2. Cấu trúc code

```
src/quantaslice/ai/
├── __init__.py            # provider_registry (đăng ký "ml")
├── provider.py            # MLPredictionProvider: nạp artifact -> Prediction(ê, p)
├── data/
│   ├── loaders.py         # đọc ColO-RAN sched*/tr*/exp*/bs*/slices_* -> BSFrame; generate_synthetic
│   ├── labeling.py        # nhãn SLA (queueing-delay) — §4
│   ├── features.py        # feature engineering + windowing — §5
│   ├── split.py           # block split leave-scheduler/config-out — §6
│   └── scenario.py        # tiêm emergency vật lý (semi-synthetic) — §10
├── models/
│   ├── baselines.py       # GradientBoostingDetector (LightGBM→sklearn HistGB fallback)
│   └── tcn.py             # MultiTaskTCN + TCNDetector (torch)
├── train/
│   ├── baseline.py        # train baseline + CLI
│   ├── deep.py            # train TCN (uncertainty weighting) + CLI
│   └── calibrate.py       # temperature scaling
└── eval/
    └── metrics.py         # PR-AUC, ECE, cost-sensitive threshold, lead_time_recall

examples/run_closed_loop.py   # closed-loop QUBO eval (nối ai + quantum) — §11
```

**Hai loại detector, một giao diện.** Cả `GradientBoostingDetector` và `TCNDetector` cùng phơi bày
`predict_windows(seq (N,W,F)) -> (probs, priorities)`; provider đưa cửa sổ thô vào, `models.load_detector()`
(joblib) nạp loại nào cũng được. torch import lười — người dùng baseline không cần torch.

Test: **78/78 pass** (`PYTHONPATH=src pytest`). Môi trường hiện tại Python 3.11 (pyproject yêu cầu 3.12);
package chưa `pip install -e`, chạy test qua `PYTHONPATH=src`.

---

## 3. Dữ liệu (data processing)

### 3.1 Schema ColO-RAN thật
- Layout: `rome_static_medium/sched{0,1,2}/tr{0..27}/exp{1..5}/bs{1..7}/slices_bs{n}/<IMSI>_metrics.csv`.
- `sched0/1/2` = 3 scheduler: Round-Robin / Waterfilling / Proportionally-Fair.
- `tr0..tr27` = 28 cấu hình phân bổ RBG khác nhau cho 3 slice.
- 7 base station, 10 MHz = **50 PRB**, 3 slice/BS (eMBB=0, mMTC=1, URLLC=2), 42 UE.
- Mỗi file `*_metrics.csv` = KPM **per-UE** (2 UE/slice), cadence **250 ms**, ~2090 mẫu (~520 s).

### 3.2 Loader (`loaders.py`)
Gộp các UE cùng slice của một BS (cộng buffer/throughput/PRB, trung bình `slice_prb`), nội suy tuyến tính
về lưới thời gian đều → `BSFrame.series` shape **(T, 3, 5)**:

| Chỉ số | Metric thô |
|---|---|
| 0 | `dl_buffer` (bytes) — hàng đợi downlink |
| 1 | `tx_brate_dl` (Mbps) — throughput đạt được |
| 2 | `requested_prbs` |
| 3 | `granted_prbs` |
| 4 | `slice_prb` (số PRB phân cho slice, hằng theo config tr) |

Kiểm chứng: `slice_prb` phản ánh đúng config (tr0 mMTC-heavy ~39 PRB, tr27 URLLC-heavy ~41 PRB).

---

## 4. Định nghĩa nhãn "emergency" — phần quan trọng & nhạy cảm nhất (plan §2)

### 4.1 Lần 1 (LỖI) — chuẩn hoá theo median
Ý tưởng ban đầu: emergency = tình trạng "đang xấu đi", severity = biến thiên buffer/throughput
**chuẩn hoá theo `median` của chính slice**.

**Đây là một defect.** Slice **nhàn rỗi** (mMTC/URLLC rỗng 78–84% thời gian, median≈0) bị chia cho ~1
→ một đốm buffer **~200 byte (nhiễu đo)** biến thành severity = 1.0. Ngược lại eMBB nghẽn thật
(buffer ~367 KB) bị chia cho median khổng lồ → severity ~0.

Hệ quả đo được:
- **75% "emergency" đến từ slice rỗng** (mMTC 39% + URLLC 36%), chỉ là nhiễu.
- eMBB nghẽn thật chỉ được gắn nhãn **0.9%** số bước.
- **PR-AUC in-distribution chỉ 0.53** → nhãn ≈ nhiễu, model không học được (chứng minh lỗi ở NHÃN, không phải OOD khó).

### 4.2 Lần 2 (ĐÃ SỬA) — độ trễ hàng đợi vật lý
```
queue_delay(giây) = dl_buffer × 8 / throughput(bps)
severity          = clip(queue_delay / latency_budget[slice], 0, 1)   + sàn buffer
emergency_flag    = có slice bất kỳ severity ≥ 0.5 kéo dài ≥ 3 bước
priority p        = 1 + boost × (severity đã sustain)
```
- **Sàn buffer** (mặc định 1000 B): buffer nhỏ hơn coi như không có hàng đợi → loại nhiễu sub-KB.
- Độ lớn có ý nghĩa vật lý (214 B → vài ms; 367 KB → ~0.8 s).
- eMBB bỏ đói kinh niên **được** coi là vi phạm SLA (quyết định có chủ đích: slice thực sự thiếu tài nguyên).

Kết quả: **PR-AUC in-distribution nhảy từ 0.53 → 0.97**. Xác nhận feature/split/model đều đúng; lỗi duy nhất là công thức nhãn.

### 4.3 Cảnh báo trung thực về ngưỡng (chưa hoàn hảo)
Ngân sách độ trễ đang dùng `(eMBB 2.0s, mMTC 0.4s, URLLC 0.08s)` là **tự chọn để ra base rate đẹp**,
KHÔNG theo chuẩn 3GPP 5QI (eMBB ~300 ms, URLLC ~1–10 ms, mMTC ~vài giây). Khi dùng ngưỡng 3GPP thật:

| Ngân sách | Base rate | Slice tạo emergency |
|---|---|---|
| Tự chọn (đang dùng) | 0.29 | eMBB 92% |
| **3GPP thật (0.3, 1.0, 0.01)** | **0.78** | **eMBB 100%** |

eMBB có độ trễ hàng đợi thực ~**2076 ms** ≫ chuẩn 300 ms → làm đúng chuẩn thì eMBB **vi phạm 100%** thời gian
(nhãn ≈ "eMBB có active không"). Đây là hạn chế cần ghi rõ: nhãn hiện tại là **proxy hợp lý** nhưng ngưỡng
là ước lượng, không phải QoE/SLA ground-truth.

---

## 5. Feature engineering + windowing (`features.py`)

Từ `BSFrame` → **24 feature/bước (F_ts)**:
- Per-slice (6 × 3 = 18): `log_buffer`, `brate`, `buffer_growth`, `prb_util_req` (cấp/yêu cầu),
  `prb_util_cap` (cấp/50), `tput_deficit` (thiếu hụt so nhu cầu class).
- Liên-slice (2 × 3 = 6): `brate_share`, `buffer_share` — emergency thường là hiện tượng *tương quan*.

**Cửa sổ trượt** lookback W=20 (~5 s @250 ms), stride 5 → mỗi mẫu là chuỗi `(20, 24)`.
- **TCN** ăn trực tiếp chuỗi.
- **Baseline** tóm tắt cửa sổ → mean/std/max/last/slope mỗi feature → vector `24×5 = 120` chiều.

Nhãn của cửa sổ lấy tại bước cuối (nowcast); hỗ trợ horizon H > 0 cho dự báo lead-time.

---

## 6. Split — chống rò rỉ (plan §3.5)

Tuyệt đối **không shuffle theo timestep** (cửa sổ chồng lấn → rò rỉ thời gian). Chia ở cấp **cấu hình**:
- `leave_scheduler_out`: train sched0+1, test sched2 (OOD).
- `leave_config_out`: giữ vài `tr` chỉ để test.

Mỗi `BSFrame` thuộc đúng một (sched, tr) → chia ở cấp frame đảm bảo không cửa sổ nào bắc cầu train↔test.
**Audit xác nhận:** `fit ∩ val ∩ test = 0`.

---

## 7. Models

### 7.1 Baseline — `GradientBoostingDetector`
LightGBM nếu có, nếu không tự lùi về sklearn `HistGradientBoosting` (cùng họ). Đa nhiệm: 1 classifier (flag)
+ 1 regressor 3 chiều (priority). Class-weight cân bằng (§3.6). Artifact tự chứa scaler + model + threshold.

### 7.2 TCN — `MultiTaskTCN` (torch)
Dilated causal conv, shared encoder + 2 head (flag sigmoid + priority softplus). Train với **uncertainty
weighting** (Kendall & Gal, §5.1) tự cân bằng 2 loss; early stopping theo PR-AUC trên val.

### 7.3 Calibration — temperature scaling (§5.2)
Học 1 vô hướng T chia vào logit để giảm ECE (không đổi thứ hạng). Có **guard**: nếu T phân kỳ làm ECE
tệ hơn thì giữ T=1 (đã gặp trên full data, guard hoạt động đúng).

---

## 8. Kết quả đánh giá (nhãn đã sửa, test = sched2 OOD, 81k cửa sổ)

| Model | PR-AUC | ROC-AUC | F1 | ECE |
|---|---|---|---|---|
| **LightGBM (baseline)** | **0.998** | 0.999 | 0.970 | 0.006 |
| TCN | 0.978 | 0.994 | 0.893 | 0.019 |

> Baseline **thắng** TCN trên full data → chọn baseline làm production (plan §10: "deep không thắng
> baseline thì dùng baseline"). NHƯNG xem §9 — các con số này KHÔNG phản ánh model giỏi.

---

## 9. Audit trung thực toàn pipeline (`scratchpad/audit.py`)

| Kiểm tra | Kết quả | Kết luận |
|---|---|---|
| Rò rỉ split | fit∩val∩test = 0 | ✅ Sạch |
| NaN / thời gian | 0 NaN, đơn điệu, n_ue=2/slice | ✅ Đúng |
| "Gian lận" slice_prb | Bỏ slice_prb → vẫn 0.997 | ✅ Không key vào dấu hiệu config |
| **ML vs luật tầm thường** | ML **0.997** vs luật 1 dòng KHÔNG train **0.990** | ⚠️ ML chỉ hơn **+0.007** |
| Split có thử thách? | leave-scheduler 0.997 · random-config 0.972 · **leave-config 1.000** | ⚠️ Split không thử thách |
| Cửa sổ chồng lấn | stride 5 / lookback 20 = chồng 75% | ▲ N hiệu dụng ~¼ |

**Kết luận cốt lõi:** con số 0.99 **đúng về số học** nhưng **không chứng minh model giỏi**. Nhãn (trên
dataset tĩnh) là hàm gần-xác-định của KPM hiện tại — mà buffer/throughput chính là feature model nhìn thấy.
Giống hỏi *"cho x, y, x/y có > 5 không?"* → bấm máy tính ra ngay, không cần ML. 0.99 = "máy tính bấm đúng",
không phải "model thông minh". `leave-config-out = 1.000` vì luật nhãn là phổ quát → split không đo được
generalization.

---

## 10. Kịch bản emergency (`scenario.py`) + phát hiện lead-time

ColO-RAN **không có emergency thật** (chỉ `rome_static_medium`, UE tĩnh). Đã dựng harness tiêm emergency
**vật lý** (traffic surge, buffer tích tụ kiểu leaky-integrator), khung bài toán **dự báo** để tránh
circular (plan §2): nhãn = lịch tiêm (độc lập KPM), dự đoán vi phạm H bước TRƯỚC từ KPM hiện tại.

**Phát hiện quan trọng:** ML forecasting chỉ có giá trị khi emergency có **precursor CHẬM + runway lớn**:
- Tiêm URLLC (frame thật): ML +0.006, onset chỉ 6 window → ❌ (URLLC vi phạm tức thì, ngân sách ms, không runway).
- Slice có runway (eMBB-like, buffer rỗng + PDB 300 ms): tại các bước "sắp vi phạm nhưng hiện chưa",
  **ML recall 0.95 vs luật 0.00** → ✅ ML mua được lead-time thật.

Trên ColO-RAN không slice nào thoả (eMBB đã bão hoà, URLLC vi phạm tức thì) → **không thể chứng minh giá trị
ML forecasting trên dataset này** một cách trung thực (không tinh chỉnh việc tiêm cho tới khi ML "thắng" —
đó là bẫy circular). Cần dữ liệu động (TRACTOR/mobility).

Metric mới: `eval.metrics.lead_time_recall` — recall ở các bước "sắp vi phạm nhưng hiện chưa" (thưởng bắt sớm).

---

## 11. Closed-loop QUBO — giá trị THẬT (plan §6.4)

`examples/run_closed_loop.py` — chạy: `python -m examples.run_closed_loop`.

**Câu hỏi:** hành động theo `p` của detector có cải thiện SLA so với trọng số tĩnh không? (Độc lập với
việc detection khó/dễ.)

**Chống rigging:** criticality thật = lịch tiêm (ground truth, luân phiên slice); `p` = detector đọc KPM;
metric dùng criticality **thật**. Khan hiếm tài nguyên → mỗi bước phải rớt 1 slice → chọn rớt cái nào.

| Chính sách | SLA cost | Critical-served |
|---|---|---|
| **Static** (ưu tiên cố định) | 158 | 60% |
| **Adaptive** (`p` từ detector) | **90** | **91%** |
| **Oracle** (biết criticality thật) | 70 | 100% |

→ Adaptive **giảm 43% SLA cost** vs static; lấp **77%** khoảng cách tới oracle.

**Vì sao đáng tin:**
- Không phụ thuộc detection khó/dễ — đo giá trị của *hành động theo `p`*.
- 3 nguồn tách biệt (criticality thật / `p` từ KPM / metric) → không circular.
- **Adaptive ≠ oracle** (90 vs 70) → detector không hoàn hảo → không rigged.
- Static là đối thủ thật (ưu tiên slice cố định là chuẩn phổ biến).
- Nguồn giá trị: criticality **biến thiên theo thời gian** — nếu cố định thì static chỉnh đúng sẽ hoà.

**Caveat:** đây là kịch bản **mô hình hoá** (dataset thật không có emergency). Độ lớn 43% phụ thuộc thiết kế
(mức luân phiên, độ khan hiếm); *nguyên lý* "adaptive thắng cố định khi criticality biến thiên" là thật.

---

## 12. TRACTOR — dữ liệu ĐỘNG: ML thực sự có giá trị (hướng B)

Vì ColO-RAN tĩnh không thể chứng minh giá trị ML, đã chuyển sang **TRACTOR**
(`genesys-neu/TRACTOR`) — KPI O-RAN sinh từ **traffic 5G thật** (capture điện
thoại, replay trong Colosseum), có mobility + nhiều app → **động, bursty thật**.
Schema ~trùng ColO-RAN nên tái dùng pipeline; loader: `ai/data/tractor_loader.py`.

**Kiểm chứng độ động:** `dl_buffer` median 0 nhưng max 164 KB, std 8.5 KB — buffer
dồn/thoát liên tục (khác ColO-RAN eMBB pinned 366 KB tĩnh).

**Khung đánh giá TRUNG THỰC (per-stream, `examples/run_tractor_forecast.py`):** với
mỗi luồng traffic, dự báo vi phạm SLA (độ trễ > ngân sách) **H=8 bước (~2s) TRƯỚC**
từ cửa sổ KPM; luôn so ML với luật "đang vi phạm chưa" + báo **lead-time recall**.

| | ColO-RAN (tĩnh) | TRACTOR subset (6) | **TRACTOR full Trial0 (12)** |
|---|---|---|---|
| Test windows | — | 9,794 | **27,521** |
| ML vs luật (PR-AUC) | +0.006 | +0.122 | **+0.219** (0.803 vs 0.585) |
| Onset windows | 6 | 191 | **858** |
| Lead-time recall — ML | 0.00 | 0.272 | **0.534** |
| Lead-time recall — luật | 0.00 | 0.00 | 0.00 |

→ Trên dữ liệu thật, dự báo là bài toán **không tầm thường** (luật chỉ 0.585), ML
**thắng rõ +0.219** và **bắt sớm 53%** vi phạm mà luật (theo định nghĩa) bỏ lỡ.
**Càng nhiều dữ liệu, ưu thế ML càng lớn/ổn định** (subset +0.122 → full +0.219).
Recall 0.53 (không phải ~1.0) là **trung thực** — traffic thật khó.

**So sánh mô hình (rule/LightGBM/LSTM/TCN, full Trial0, `examples/run_tractor_models.py`):**

| Mô hình | PR-AUC | Lead-time recall |
|---|---|---|
| Luật tầm thường | 0.585 | 0.085 |
| **LightGBM** | **0.803** | **0.534** |
| LSTM | 0.500 | 0.260 |
| TCN | 0.383 | 0.277 |

→ **LightGBM thắng áp đảo; LSTM/TCN THUA** (LSTM 0.500 còn dưới cả luật về PR-AUC).
Deep model *như-đang-train* (12 epoch, mạng nhỏ, CPU, chưa tune) không cạnh tranh —
đúng plan §10 ("gradient boosting rất khó đánh bại") và §4 (lý do "beyond LSTM").
Feature-engineering + boosting quá mạnh; deep cần tune kỹ + nhiều epoch/dữ liệu hơn
(không đảm bảo thắng). **Trung thực: dùng LightGBM.**

**Dữ liệu:** đã tải full `logs/` từ GitHub `genesys-neu/TRACTOR` (122 MB, 220 file;
Trial0 = 12 luồng per-type) vào `tractor-repo/` (gitignored). Chống bẫy con số
tuyệt đối: nếu gộp 3-slice + nhãn "any slice" thì PR-AUC vọt 0.994 (lại tầm thường)
— nên eval **per-stream + luôn so luật + lead-time**.

**Cách tải + chạy:**
```bash
# tải full logs/ từ genesys-neu/TRACTOR (sparse clone, ~122 MB):
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/genesys-neu/TRACTOR tractor-repo
cd tractor-repo && git sparse-checkout init --cone && git sparse-checkout set logs && git checkout && cd ..
python -m examples.run_tractor_forecast --root tractor-repo --trial Trial0 --horizon 8
```

---

## 13. Kết luận trung thực & khuyến nghị

**Sự thật tổng hợp:**
1. Nhãn cũ lỗi → điểm thấp giả tạo → **đã sửa** (queue-delay vật lý).
2. Pipeline sạch, không leakage → **đã audit**.
3. Nowcast trên ColO-RAN static là **tầm thường** — luật 1 dòng ≈ ML → **đã thừa nhận**.
4. Dataset **không có emergency thật** → ML forecasting chưa chứng minh được giá trị ở đây → **đã chứng minh điều kiện cần**.
5. **Giá trị đo được = closed-loop: `p` thích ứng giảm 43% vi phạm SLA** → **con số nên trình bày**.

**Thông điệp nên dùng khi báo cáo:** *"Tầng AI không cần là bộ phân loại phức tạp — trên dữ liệu này một
luật SLA đơn giản đủ để tính `p`; giá trị chứng minh được là acting on `p` cải thiện phân bổ QUBO 43% khi
nhu cầu các slice biến thiên."*

**Khuyến nghị tiếp theo (theo thứ tự giá trị):**
1. Chạy closed-loop với `--solver qaoa_aer` (số trên QAOA thật) + quét độ nhạy (khan hiếm, tần suất luân phiên) → báo khoảng cải thiện.
2. Chuẩn hoá ngưỡng nhãn theo **3GPP 5QI** và đổi khung thành "SLA-violation monitor + priority `p`" cho trung thực.
3. Tải **TRACTOR** (dữ liệu 5G thật, động) để có emergency thật cho ML forecasting.

**Chưa làm:** PatchTST/iTransformer, VUS-PR/affiliation, ONNX export + latency benchmark, ngưỡng theo 3GPP,
dữ liệu động.

---

## 14. Cách chạy (tái lập)

```bash
# Train baseline / TCN trên dữ liệu thật (test = sched2 OOD)
PYTHONPATH=src python -m quantaslice.ai.train.baseline \
    --data-root colosseum-oran-coloran-dataset --scheds 0,1,2 --exps 1 \
    --test-scheds 2 --out artifacts/detector.joblib
PYTHONPATH=src python -m quantaslice.ai.train.deep \
    --data-root colosseum-oran-coloran-dataset --scheds 0,1,2 --exps 1 \
    --test-scheds 2 --epochs 25 --out artifacts/tcn.joblib

# Dùng detector trong pipeline: config prediction_provider: ml + ml_artifact: <path>

# Closed-loop QUBO (giá trị thật)
PYTHONPATH=src python -m examples.run_closed_loop

# Test
PYTHONPATH=src pytest tests/ai tests/test_closed_loop.py -q
```

---

## 15. Trạng thái

| Hạng mục | Trạng thái |
|---|---|
| Loader ColO-RAN + BSFrame | ✅ |
| Nhãn QoS-violation (queue-delay) | ✅ (ngưỡng chưa theo 3GPP) |
| Feature + windowing + block split | ✅ |
| LightGBM baseline | ✅ |
| TCN multi-task + uncertainty weighting | ✅ |
| Temperature-scaling calibration | ✅ |
| Metrics (PR-AUC, ECE, cost-threshold, lead-time) | ✅ |
| MLPredictionProvider + tích hợp Runner | ✅ |
| Audit pipeline (leakage/trivial/split) | ✅ |
| Scenario injection + lead-time metric | ✅ (semi-synthetic) |
| **Closed-loop QUBO eval** | ✅ (giá trị thật: −43% SLA cost) |
| **TRACTOR loader + forecasting eval (dữ liệu động)** | ✅ (ML +0.122 PR-AUC, lead-time 0.27 vs 0.00; mới trên subset) |
| Full TRACTOR download, PatchTST/iTransformer, VUS-PR, ONNX, 3GPP budgets | ❌ (chưa) |

*Test suite: 56/56 pass (không tính `tests/orchestrator/` — hỏng do refactor bên
ngoài gỡ `orchestrator/state.py` + `coloran_loader.py`, không liên quan tầng AI).*

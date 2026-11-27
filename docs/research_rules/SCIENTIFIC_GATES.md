# Scientific Gates — exact contract router

Governance v2.0.0 — 2026-11-27 — documentation only. Historical research state pinned to a2a1056075f4a41e9f05115b30e2a50144b34154.

This module routes to frozen contracts, not replacement thresholds. Read exact linked sections before any related scientific decision; all old experiment-specific definitions, denominators, fixture semantics and thresholds remain in force for their own experiment. If a summary disagrees with exact source, hold and resolve. No threshold relaxation.
 
## General criteria

1. **Không lấy push *đã cấu hình* làm push *đã xảy ra*.** Ghi sự kiện thực tế trong chính **PPO transitions** dùng cho gradient. Một rollout 32 steps/env không có source push đầu nếu trigger từ sau 3–5s; M4.63 scheduler đã tạo push sớm và đo được.
2. Ghi các phân phối **đồng thời**: motion × body-frame direction × actual stance/contact phase × magnitude × episode progress × velocity/yaw × lần push thứ mấy × post-push survival. Marginal counts 6 hướng, 6 chuyển động, 3 pha **không chứng minh đã phủ hết 108 tổ hợp**.
3. Bao phủ các tình huống đứng/đi chậm/đi nhanh/đi lùi/đi ngang/xoay hai chiều, một chân chống, hai chân chống, đẩy trái/phải/trước/sau/chéo, đang phục hồi và push liên tiếp. **Độ khó là curriculum tăng dần theo kết quả**, không cố tăng cường độ tới mức mọi episode ngã và gradient không học được phục hồi.
4. Giữ **nominal/no-push control exposure** trong train và eval để tránh gait collapse. Tỷ lệ 20% hiện tại là **mục tiêu sampler**, phải kiểm tra *episode-level* và *transition-level* riêng, không đánh đồng.
5. Tách `push magnitude` là xung `Δqvel` (m/s) với lực (N), thời gian tác động/impulse, hướng trong body/world; mỗi thử nghiệm phải báo cáo cách tạo push.
6. **Đo thời gian còn lại sau push** trong rollout/training, số hành động điều khiển *thật sự* dùng sau push. Số lượng push cao nhưng thiếu tail để học recovery là không đạt mục tiêu học phục hồi. Với PPO mới warm-start, báo số bước và cập nhật optimizer, và xác định nếu restart Adam.
7. Cấu hình `easy/medium/hard` đã định nghĩa chỉ là **giả thuyết curriculum**. Không tự động nhảy sang mức khác hoặc mở pilot 20 rollouts khi chưa qua gates về gait và recovery.

## 5. Fall / Get-up / Resume — tiêu chí bắt buộc mới

**Tuyệt đối không trộn hai bài toán:** `push survival / recovery khi còn đứng` khác `tự đứng dậy từ trạng thái đã ngã`. PPO locomotion hiện tại có `termination` và auto-reset; việc học get-up đòi môi trường cho phép trạng thái sau ngã tiến triển qua nhiều bước, có dữ liệu quan sát và action hợp lệ trong các tư thế ngã.

**Đường triển khai có gate:**

- **F0 — Fall forensics:** kiểm tra điều kiện `done`, root height, body tilt, support/contact, torques, joint limits; phân loại ngã thật vs termination khác. Thu tập fallen states hợp lệ, không lưu/sử dụng reset như recovery. Chỉ đọc/đánh giá, không thay locomotion reward.
- **F1 — Get-up physics feasibility:** trong sandbox thí nghiệm riêng, thử điều khiển deterministic/trajectory hoặc chính sách get-up xuất phát từ nhiều tư thế ngã; kiểm tra quan sát/action, mô-men, contact và tự nâng thân; không nhảy tư thế. Chứng minh 1 ca sấp + 1 ca ngửa có điều khiển trước khi RL quy mô lớn.
- **F2 — Get-up learning:** huấn luyện riêng nếu F1 PASS và VRAM fits, có curriculum từ tư thế ngã ít khó đến khó, đa dạng pose/terrain/contact. Đánh giá xác suất đứng dậy, thời gian get-up, torque, va chạm và stability, không chỉ reward.
- **F3 — Integrated resume:** đẩy → ngã thật → GET_UP → REORIENT → tiếp tục lệnh; **không auto-reset giữa các pha** trong các episode được tính thành công. Giữ command goal qua quá trình đứng dậy, có watchdog/thời gian thất bại được ghi riêng. Mỗi tập phải ghi trạng thái chuyển tiếp và mục đích termination/abort.
- **F4 — Robustness / sim-to-sim:** MJX GPU → MuJoCo CPU, cùng trạng thái đầu/lệnh/kịch bản khi khả thi; kiểm tra sensitivity của ground friction/PD/contact và rõ ràng domain randomization, không suy thành hardware validation.

**Định nghĩa thành công get-up (đề xuất, phải hiệu chỉnh bằng baseline kiểm thử):** từ fallen pose thật, sử dụng torque/action liên tục trong mô phỏng, robot lên upright, đứng ổn định trong một khoảng kiểm tra cố định, sau đó đạt tracking command trong cửa sổ đánh giá riêng. Ba sự kiện `(GET_UP_SUCCESS, STABILIZATION_SUCCESS, RESUME_SUCCESS)` phải được ghi **riêng**; chỉ ghi hoàn thành chu trình khi cả ba đạt và không có teleport/reset.

## 6. Hệ đo và so sánh — bắt buộc cho nghiệm thu

**Training-side:** count push trong PPO transitions, unique episodes có push, category *joint histogram*, thực tế phase target vs phase actual, push mạnh/yếu, event exposure, số bước sau push trước reset/rollout, done theo nguyên nhân, VRAM peak, checkpoint SHA, finite gradients/weights, optimizer state.

**Nominal:** survival horizon, forward/backward/lateral/yaw command tracking RMSE, chiều cao, tilt, joint/foot motion, tốc độ chuyển đổi lệnh, hội tụ khi đứng yên. Không kết luận từ `eval reward` 2 env.

**Push:** ghép **ON/OFF từ cùng full pre-push state trong từng policy**; same evaluation reset seeds + command + push directions/magnitudes/timings giữa policy. Ghi số lần ngã, first-fall time, sống sót 2s/5s/to horizon, matched-OFF recovery 2s/5s, thời gian quay về lệnh, RMSE **cửa sổ bằng nhau**; nếu tử vong trước cửa sổ, ghi censored/fail, không so RMSE 50 bước với 700 bước. Khác policy có thể có full physical state khác dù reset seed cùng.

**Get-up:** tách initial fallen pose distribution và push-induced fall distribution; tỷ lệ tự đứng dậy theo từng pose/dose/command, time-to-upright, stable-time, torque/collision cost và **resume tracking**. Cần đánh giá failure/timeout rõ ràng và không auto-reset; một fall là một thử nghiệm get-up thật, không phải khởi tạo đứng lại.

**Khoa học:** các reset seeds development không độc lập giữa các kịch bản lặp; không dùng 72 lượt từ ba seeds làm 72 mẫu độc lập. Khi tuyên bố cải thiện, cần freeze tập held-out độc lập, đối chứng cùng ngân sách/seed và có nguồn paper rõ. Phân biệt **technical PASS**, **exposure PASS**, **nominal gait PASS**, **robustness PASS** và **get-up/resume PASS**; một loại không suy ra loại khác.

 
## Active family H4/FFE
- Official CTRL primary, archived M12 descriptive; matched candidate-only DEV: 12 groups/60 logical trajectories/max44808 physical/12 resets; seeds 7331/7357/7601/7613; stand/forward_fast/turn_left, L/R, nominal/single/double, pushes 300/.50 and 335/.35, deterministic DR OFF. +2/8 standing double survival AND recovery5, plus unchanged nominal/moving/safety gates. No promotion or holdout for FFE-V1. Exact contract [v1.2.294 L3792–3805](https://github.com/nguyenbui5701/G1_PAPER_REPRO/blob/a2a1056075f4a41e9f05115b30e2a50144b34154/G1_RESEARCH_RULES.md#L3792-L3805), [v1.2.294 L3980–3989](https://github.com/nguyenbui5701/G1_PAPER_REPRO/blob/a2a1056075f4a41e9f05115b30e2a50144b34154/G1_RESEARCH_RULES.md#L3980-L3989); original decision logic in frozen evaluator, not reconstructed from prose.
- FFE material gates: half-cell >=3 each, chunk >=1, nominal >=916743, broad floors 1256/1262/1360/1296/1344, legacy conservation; exact prospective proposal and original constraints [v1.2.294 L3778–3791](https://github.com/nguyenbui5701/G1_PAPER_REPRO/blob/a2a1056075f4a41e9f05115b30e2a50144b34154/G1_RESEARCH_RULES.md#L3778-L3791), training design [v1.2.294 L3504–3525](https://github.com/nguyenbui5701/G1_PAPER_REPRO/blob/a2a1056075f4a41e9f05115b30e2a50144b34154/G1_RESEARCH_RULES.md#L3504-L3525), integration and receipt [v1.2.294 L3712–3777](https://github.com/nguyenbui5701/G1_PAPER_REPRO/blob/a2a1056075f4a41e9f05115b30e2a50144b34154/G1_RESEARCH_RULES.md#L3712-L3777). Preserve exact signed actual committed rows and no mutation on invalid batch.
- Earlier gate families: use EVIDENCE_INDEX by task/section; retrieve complete applicable range and referenced immutable source. No reuse of an unrelated threshold or inference from a single metric. Branch B gate definitions preserved but not authorized.

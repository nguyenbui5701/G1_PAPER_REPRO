# G1 RESEARCH RULES — PPO Push Robustness → Fall Recovery → Get-Up → Resume

**Version:** 1.1.0  
**Ngày cập nhật:** 2026-11-21 (Asia/Ho_Chi_Minh)  
**Trạng thái:** ACTIVE — tài liệu điều hành bắt buộc cho các task tiếp theo  
**Repository:** `https://github.com/nguyenbui5701/G1_PAPER_REPRO`  
**Nhánh làm việc:** `experiment/b1-hardware-scaled` — **KHÔNG sửa `main` hoặc tag/checkpoint baseline**  
**Thiết bị mục tiêu:** RTX 3060, 12 GiB VRAM; MuJoCo Playground 2025 / MJX / PPO.

> **Quy tắc số 0:** Trước MỌI task phân tích, sửa code, chạy mô phỏng, huấn luyện hoặc viết báo cáo, đọc toàn bộ file này cùng mục `CURRENT STATE` và các báo cáo liên quan. Kết thúc task bằng bằng chứng, quyết định GO/NO-GO và cập nhật `CURRENT STATE`, `TASK LEDGER`, `CHANGELOG` nếu có sự thay đổi. Không suy diễn kết quả chỉ từ reward hoặc số lượng push.

## 1. North Star — mục tiêu nghiên cứu đã khóa

**G1 phải (A) thực hiện lệnh đi/đứng/xoay bình thường; (B) chịu push thực sự trong dữ liệu huấn luyện, giữ thăng bằng và phục hồi bám lệnh; (C) khi đã ngã, tự đứng dậy bằng điều khiển vận động, ổn định lại và tiếp tục lệnh trước đó.**

**Chuỗi hành vi cần chứng minh:** `FOLLOW_COMMAND → PUSHED → RECOVER_UPRIGHT` **hoặc** `FALLEN → GET_UP → REORIENT → RESUME_COMMAND`. Không được tính reset episode, teleport pose, đặt lại root `qpos`, hoặc xuất hiện robot đứng sẵn ở episode mới là một lần tự đứng dậy thành công.

**Ưu tiên nghiên cứu:** push exposure thật → push recovery → phát hiện ngã chính xác → học get-up → nối get-up với locomotion và resume. Không tiếp tục tối ưu reward cho gait bình thường như mục tiêu độc lập; vẫn phải bảo toàn gait đã đạt. Paper fidelity (source/original) và *hardware-scaled robustness extension* phải báo cáo thành hai dòng riêng.

## 2. Baseline bất biến, môi trường, provenance

- Pin MuJoCo Playground source: commit `9bcbe736177472ba35d60773e3030a36e37962b7`.
- Pin Brax PPO `train.py` SHA-256: `e20613d42bf0868a13003e0312998816cc8d3593bb5ccaf9f1d070fd09c44bd7`.
- Checkpoint tham chiếu **C**: PPO seed 1, **401,080,320** steps, SHA-256 `1ccf579ea35d4518713b00ea20d77db8463bc469b7de903b8f10f457b3dc90b8`.
- Checkpoint push-coverage **P (M4.63 easy smoke)**: **262,144 bước PPO bổ sung từ trọng số C**, SHA-256 `cd2cebe1d1b2ef74050d366814ee8d19259939f45612db67ae60a8a58e3003f0`. Khởi tạo Adam mới; **KHÔNG phải full optimizer resume**.
- Bộ train hardware-scaled đã smoke: 8,192 env × 32 steps/env = 262,144 steps/rollout, batch 1,024, 8 minibatches, 5 updates/batch ⇒ **40 minibatch optimizer updates/rollout**. Source/paper-exact `num_envs=32,768` không phù hợp GPU này; không gọi bản scaled là paper-exact.
- Tiếp tục dùng venv JAX/CUDA và các `PYTHONPATH` overlay đã chứng minh chạy được; `nvcc_compat.apply()` phải xảy ra **trước khi import JAX**. Không tự ý `pip install`/đổi package, source, PD, reward, phase, termination, action scale, randomizer hay mạng PPO.
- Không tạo training seed mới khi chưa có lý do được ghi rõ. Các **reset seeds 7301, 7331, 7357** đã dùng để development, **không gọi là final holdout**. Tái sử dụng evaluator, danh sách seed/command/push và các checkpoint đã có; không bắt người dùng chạy lại thí nghiệm đủ bằng chứng.
- Sửa trực tiếp code hiện hữu khi hợp lý, tránh mọc nhiều script. Những thay đổi để có get-up chỉ được thực hiện trong **môi trường/nhánh thí nghiệm riêng**, có diff, ablation và mô tả rõ, không sửa âm thầm code nguồn paper.

## 3. Các trạng thái và biên nghiên cứu bắt buộc

| Trạng thái | Mục tiêu & dấu hiệu | Điều KHÔNG được giả định |
|---|---|---|
| `UPRIGHT_TRACKING` | Đứng, đi/lùi/đi ngang, xoay theo command | Reward tốt = gait tốt |
| `PUSH_RECOVERY` | Push làm lệch trạng thái; robot phục hồi và tiếp tục lệnh | Không ngã = đã phục hồi bám lệnh |
| `FALL_DETECTED` | Xác định ngã từ tín hiệu đo phù hợp (root height/orientation/contact, không chỉ một ngưỡng tùy tiện) | `done=true` luôn có nghĩa đã ngã; terminate có thể vì nguyên nhân khác |
| `GET_UP` | Chính policy/controller tạo hành động nâng robot từ **trạng thái đã ngã** | Reset/teleport hoặc khởi tạo tư thế đứng là get-up |
| `REORIENT` | Ổn định trục thân, chân/chân chống, hướng vận động | Root height tăng một lần = đứng ổn định |
| `RESUME_COMMAND` | Nhận lại đúng command hoặc command cập nhật và thực hiện liên tục | Đứng dậy xong mà đứng im = hoàn thành nhiệm vụ |

**Kiến trúc nghiên cứu ban đầu (đề xuất, chưa được xác minh):** Giai đoạn 1 giữ frozen/được huấn luyện tiếp `locomotion-PPO`; giai đoạn 2 phát triển **get-up policy/controller riêng** với fallen-state initializations đúng vật lý; giai đoạn 3 nối bằng supervisory state machine, sau đó mới thử single-policy/hierarchical RL nếu có bằng chứng. Dùng phiên bản tách biệt để tránh catastrophically forgetting gait cũ. **Không giả định checkpoint locomotion hiện tại có thể đứng dậy**: pinned joystick mặc định terminate/reset, nên phải thiết kế môi trường get-up cho phép roll out **sau ngã** và kiểm chứng đổi trạng thái thực, không phá nguồn baseline.

Các tư thế nghiên cứu get-up: ngã sấp, ngã ngửa, nằm nghiêng trái/phải, quỳ/thân thấp, nhiều góc yaw và sai lệch khớp hợp lệ. Chú ý self-collision, giới hạn khớp/mô-men, contact/ma sát, actuator dynamics và tính chính xác mô phỏng. Thử nghiệm trên MuJoCo/MJX trước; không suy khả năng triển khai phần cứng thật từ kết quả mô phỏng.

## 4. Huấn luyện push thật — kiểm soát exposure và chất lượng học

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

## 7. Quy trình bắt buộc cho MỖI task/thử nghiệm

**Trước task:**

1. Đọc file này và trạng thái cuối `CURRENT STATE`; ghi `task_id`, giả thuyết, năng lực mục tiêu (`PUSH`, `FALL`, `GET_UP`, `RESUME`), input files, branch/commit, checkpoint SHA, source SHA, venv/VRAM, seed và đánh giá **GO/NO-GO**.
2. Tìm/mở các script, evaluator, log/report đã có; **ưu tiên sửa tối thiểu file cũ**; không chạy lại bằng chứng đã đủ. Xác định một biến độc lập cần thay, đối chứng và tiêu chí dừng trước khi xem kết quả.
3. Với mọi thay đổi reward/termination/PD/action/observation/model: **STOP mặc định**. Chỉ thực hiện ở experimental get-up environment khi có biện minh, diff, kế hoạch ablation, test, và không ảnh hưởng baseline/pinned source.

**Trong task:**

4. Theo thứ tự `static inspect → plan → JAX/CPU probe → GPU smoke → evaluation fixed seeds → pilot` và không bỏ qua các gate. Chỉ một GPU job nặng cùng lúc, theo dõi VRAM và lỗi/NaN. Bất cứ lỗi import / NVCC / state shape / Orbax nào phải sửa tại đúng tầng, ưu tiên mã cũ, không tạo chuỗi script workaround mới.
5. Không train mới hay nâng difficulty chỉ để lấy reward tốt. Thống kê exposure phải đo trên dữ liệu PPO. Mọi checkpoint/output dùng timestamp và tạo mới, không overwrite artifacts nguyên gốc; log cả stderr/stdout.
6. Nếu xuất hiện NO-GO (OOM, dữ liệu không đủ, SHA mismatch, nonfinite, suy giảm nominal rõ, phase/push không xuất hiện, sai pairing, get-up bằng reset), giữ file lỗi và báo cáo, **không diễn giải thành PASS**.

**Sau task:**

7. Đối chiếu JSON + log + các chỉ số; phân biệt `OBSERVED / INFERRED / UNTESTED`. Viết `result`, `limits`, checkpoint/hash, evidence files, quyết định **GO/NO-GO** và hành động cụ thể tiếp theo. Không khẳng định cải thiện chỉ từ training reward, push count hoặc một seed.
8. Cập nhật `CURRENT STATE`, thêm **1 dòng** vào `TASK LEDGER` và `CHANGELOG` khi có thay đổi rules/mục tiêu, bump version. Commit **chỉ file có ý nghĩa** lên nhánh `experiment/b1-hardware-scaled`, không push secrets, cache, huge checkpoint/video hoặc thay `main`/baseline; lưu SHA commit và đường dẫn báo cáo.
9. Khi trả lời người dùng: ghi rõ đã/chưa chạy, bằng chứng được kiểm tra, còn thiếu gì, lệnh chạy có sẵn, nguồn rủi ro và bước thử **nhỏ nhất đủ trả lời câu hỏi**.

### Quy tắc EDIT-FIRST — sửa file hiện có trước, chỉ tạo file mới khi cần (v1.1.0)

**Mặc định: KHÔNG tạo script, runner, audit, runbook, ZIP, Markdown hoặc artifact mới cho mỗi task.** Ưu tiên đọc lại source + script/evaluator/audit đang sử dụng và **đề xuất cho người dùng tự sửa** số ít dòng nhất tại đúng tầng xảy ra vấn đề. Một task có thể chỉ cần hướng dẫn sửa trực tiếp trong câu trả lời; không cần deliverable mới.

**Trước khi đề xuất thay đổi code, bắt buộc trình bày:** (1) tên file và vị trí/hàm/đoạn có thể tìm bằng `grep`; (2) code đang có và code thay thế hoặc unified diff ngắn; (3) lý do nhân–quả, giả thuyết và bằng chứng nguồn; (4) biến được thay và các biến cố định; (5) tác dụng phụ, độ tương thích source/checkpoint/audit; (6) lệnh syntax/unit/plan/smoke test, tiêu chí PASS/NO-GO và rollback; (7) cần chạy lại training hay chỉ re-audit kết quả cũ. **Không được mặc định sinh toàn bộ file khi một patch vài dòng đã đủ.**

**Chỉ tạo file mới** khi có nhu cầu độc lập thực sự: (a) tách môi trường get-up khỏi source locomotion gốc; (b) yêu cầu artifact có cấu trúc (định dạng báo cáo/manifest/chứng cứ không thể truyền qua chat hoặc log hiện có); (c) quy mô thay đổi lớn, rủi ro mất code, hoặc cần kiểm thử cách ly; (d) người dùng yêu cầu file. Khi tạo, giải thích tại sao sửa file cũ không phù hợp, phạm vi tối thiểu, đặt tên tái sử dụng và tránh tạo nhiều bản `*_v2`, `*_fixed`, `*_hotfix` cho cùng một chức năng. Artifact từ lần chạy do người dùng chủ động sinh (log/report/checkpoint) không bị cấm; nhưng không được phát sinh thêm file hướng dẫn/code lặp lại không cần thiết.

**Không tự ý ghi đè repo trên GitHub** chỉ để sửa source huấn luyện; mặc định gửi hướng dẫn để người dùng tự chỉnh và kiểm thử. Ngoại lệ: người dùng đã yêu cầu **cập nhật rules hiện hành**, hoặc xác nhận muốn assistant sửa repo. Bất kỳ thay đổi nào phải giữ baseline pin và có đường rollback. Nếu phát hiện lỗi giả định/mô tả từ task trước, đính chính công khai trong CURRENT STATE/TASK LEDGER, không tiếp tục dựa trên kết luận cũ.

### Mẫu biên bản task (copy vào báo cáo hoặc phần TASK LEDGER rút gọn)

```text
TASK_ID: M4.xx
OBJECTIVE: PUSH | FALL | GET_UP | RESUME
HYPOTHESIS:
BASELINE / CHECKPOINT_SHA:
SOURCE_COMMIT / BRAX_SHA / BRANCH:
CHANGE (independent variable):
CONTROLS (fixed):
SEEDS / COMMANDS / PUSH / FALL_POSES:
BUDGET / GPU_GATE / STOP_CRITERIA:
FILES / LOGS / REPORT_SHA:
RESULT: OBSERVED | INFERRED | UNTESTED
NOMINAL_GAIT / PUSH_EXPOSURE / RECOVERY / GET_UP / RESUME:
DECISION: GO | NO-GO | CONDITIONAL
NEXT_MINIMUM_EXPERIMENT:
RULES_VERSION_USED:
```

## 8. CURRENT STATE — cập nhật theo bằng chứng mới nhất (2026-11-21)

**M4.63 push-exposure GPU Smoke:** `M463_COVERAGE_SMOKE_PASS`; PPO train từ trọng số C với Adam reset, đã quan sát **6.503** push events thực trong training transitions; actual single support **3.454**, double support **3.045**, no-foot **4**, 6/6 hướng và 6/6 nhóm motion có exposure; VRAM peak **9.018/12.288 MiB**. **Lưu ý:** chỉ chứng minh exposure; không chứng minh learned get-up hay recovery. Eval reward nguồn 2-env giảm đáng kể, vẫn phải giữ cảnh báo regression.

**M4.63 Gate 3b mới nhận:** dữ liệu `gate3b_cp_dose050_20261121_043902.json` và audit/log cùng tên. **36 cặp C/P** (3 development reset seeds × 6 motion × 2 lateral directions), **72 policy cases**, `Δqvel=0,50m/s`, push ở step300, horizon1000; static checks và aggregator hoàn thành.

- **Nominal:** C/P đều có **0/18 nominal terminations** theo báo cáo; P bị gắn cảnh báo **XY tracking RMSE tăng hơn 5%** tại `(seed 7357, turn_right)` (`C=0,160257`, `P=0,171429 m/s`). Chỉ số 5% là ngưỡng sàng lọc đề xuất, chưa chứng minh regression thống kê.
- **Push survival:** C **26/36**, P **26/36** sống đến horizon. So sánh ghép cặp: cả hai sống **25**, P thắng C **1**, C thắng P **1**, cả hai thất bại **9**.
- **Recovery so với OFF:** ≤2s: C **1/36**, P **3/36**; ≤5s: C **19/36**, P **18/36**. Kết quả mixed; **chưa chứng minh P robust hơn C**.
- Đây là **MJX GPU exploratory**, không phải MuJoCo CPU sim-to-sim hay paper-exact; 3 seeds phát triển không thành 36 mẫu độc lập. SHA raw result `fca23505c79fea78cde5323466812418afad458b831a3683c44f274112b022fc`.

**DECISION:** M4.63 exposure PASS và Gate3b data PASS; **LONG PPO PILOT CHƯA ĐƯỢC PHÊ DUYỆT** vì robustness chưa rõ và có nominal screening flag. Không tự động chạy M4.63 dose0.75 trước khi kiểm tra trường hợp fail tại 0.50 và làm rõ tail-of-rollout/recovery learning. **GET_UP/RESUME hiện NOT IMPLEMENTED/NOT TESTED.**

**M4.64 offline update (2026-11-21):** Đã kiểm toán lại nguyên vẹn raw Gate3b với SHA `fca23505c79fea78cde5323466812418afad458b831a3683c44f274112b022fc` bằng `m464_failure_audit.py` (36 cặp). Hai nhánh khác nhau về sống sót: `(7331, stand, push right)` C sống/P chết ở step403; `(7331, turn_left, push left)` P sống/C chết ở step373. Cả hai chết 9 ca, cả hai sống 25 ca. Nominal screening `(7357, turn_right)` của P tăng XY RMSE 6.97%. Chưa xác định *cause of done* từ các JSON tổng hợp; không gọi mọi termination là fall thật.

**M4.64 tail-of-rollout:** Đã xác minh từ Brax `acting.actor_step`/`ppo/train.py` rằng rollout PPO lấy `nstate.info` làm transition extras và giữ `env_state` qua nhiều rollout. Trong *smoke đầu tiên* (32 steps/env, push đầu ở age12..31) chỉ có **tối đa 19 action sau push = 0.38s** để policy phản ứng. File `m464_tail_patch.py` cung cấp bản vá guarded cho `m463_instrument_brax.py` để đo các action này trên đúng PPO transitions; `m464_tail_audit.py` kiểm tra hậu kỳ. **5/5 test logic offline PASS, GPU instrumented T0 NOT RUN**. Chưa được mở 20-rollout pilot; có thể theo dõi tail sau khi T0 PASS. F0/F1 được thiết kế theo pinned `_get_termination()` thực tế (torso lật, foot/shin contact, nonfinite), **chưa thu fallen-state snapshot, chưa có GET_UP/RESUME code**. Không thể coi 5/10/20 checkpoints từ các runner weights-only restart là optimizer continuation; cần true-state resume gate trước khi claim training liên tục.

**M4.64-T0 audit hotfix (2026-11-21):** Báo lỗi `M464_MISSING_TRANSITION_METRIC_training/m463_events_with_1_post_actions_per_rollout` sau khi user gọi `m464_tail_audit.py`. Đã đối chiếu hai script gốc: `m464_tail_patch.py` phát ra `...events_with_1_post_action_per_rollout` (số ít), trong khi audit đòi `...post_actions...` (số nhiều). Sửa **chỉ audit** để chấp nhận hai biến thể, từ chối số liệu mâu thuẫn; chạy 5/5 test gốc + test alias offline PASS. SHA256 của bản `m464_tail_audit.py` sửa: `ff646a760840e70d5e769c435d878676b383aec2658ddd9382289b7eb4a9d865`. **Chưa nhận report/log T0 từ user, chưa xác minh T0 GPU hoặc recovery; không rerun training để chữa tên metric.** NEXT: chạy lại audit trên report.json M4.64-T0 hiện có; nếu khác lỗi, kiểm toán đúng report trước khi sửa bất cứ script PPO nào. LONG PILOT NO-GO; GET_UP NOT IMPLEMENTED.

**M4.64-T0 verified after audit hotfix (2026-11-21):** User uploaded `report(3).json` SHA256 `21cc288d162c715ba66aa053dd6b674c367fd0779871aa106c9b57221effc660` and `tail_audit_hotfix.json` SHA256 `c492bf8fda198fd814062ded2b5643f8c41f42ea9e7c6ded4f18e6168c163dcf`. Independent cross-check **PASS**: 6,501 push events in PPO transitions and 75,377 postpush action transitions, **11.59 action/push ≈0.232 s**; milestone ≥1: 6,453, ≥10: 4,261, ≥25/50/99: **0**. Peak VRAM **9,106/12,288 MiB**; source/401M pinned; 1 rollout=262,144 total steps. Eval reward 2-env source **21.036→3.766**, mean episode length **723→274.5**, warning NOT proof nominal gait regression. This T0 one-rollout checkpoint SHA `1754ff0af7f1117e3b0ac68ee347c8a842dba76f6672782402f1e367f20aaa61` differs from prior M4.63 P tested at Gate3b: do not transfer C/P test claims to T0. No training or evidence of real 2–5s recovery or get-up. **T0 TECHNICAL/EXPOSURE PASS, RESEARCH RECOVERY UNPROVEN.**

**M4.65-T1 prepared offline:** A source-preserving **5 consecutive rollouts** (1,310,720 env steps) from original 401M weights with **fresh Adam** and identical easy push config; NOT a continuation from T0 or previous M4.63 P checkpoint. New separate `m465_probe5_train.py` refuses pilot20/medium/hard, checks T0 audited report and instrumented Brax SHA before training, saves 5 sequential checkpoint states (0/1/2/3/4/5 evals). `m465_probe5_audit.py` checks >=25/50/99 postpush milestones, checkpoint SHA and per-epoch counts. **9/9 offline tests + PLAN PASS, GPU T1 NOT RUN**; no empirical robustness improvement claimed. Output artifact package and runbook available from current conversation. T1 may proceed as a *bounded diagnostic* only, while **20-rollout pilot remains NO-GO** until follow-on fixed-seed gait/push recovery comparison. F0/F1 get-up continues **design only**.

**M4.65-T1 GPU reviewed (2026-11-21; user's three files):** `m465_t1_probe5_20261121_062502.log`, `report(5).json`, `tail_audit.json`: technical/PPO exposure PASS. Five training rollouts, 1,310,720 environment steps, 32,505 pushes in PPO, 375,503 later action transitions, 11.55 action transitions per push; thresholds >=25/50/99 all **ZERO** across all five; peak 8,972/12,288 MiB. Initial 2-env source reward 21.64 and final 6.08 (small stochastic evaluator, warning only). Five saved checkpoint SHAs per report; get-up NOT IMPLEMENTED. **ROOT CAUSE CONFIRMED BY BRAX 0.12.1 SOURCE:** `m465_probe5_train.py` retains `scaled.num_resets_per_eval=1` from source hardware-scaled config while setting `scaled.num_evals=6`. Brax runs exactly 1 PPO rollout per training epoch and then executes `env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state`; therefore host resets ALL training env physical states **after each rollout**, despite optimizer weights continuing. The previous note "5 rollouts share env state" in `m465_probe5_audit.py` was WRONG for the actual T1 configuration. Absence of >=25 action tails is expected under this reset design, NOT proof physical recovery cannot be learned after proper continuous-rollout evaluation. **DECISION:** T1 technical PASS / continuous-postpush-exposure NOT TESTED / robustness NOT PROVEN; 20-rollout pilot, medium/hard/get-up training NO-GO. **EDIT-FIRST candidate for user's local re-run only:** in existing `m465_probe5_train.py`, set `scaled.num_resets_per_eval = 0` with comment and assert prior to `cfg=scaled.to_dict()`; in existing `m465_probe5_audit.py`, add a strict check that `hardware_scaled_config.num_resets_per_eval == 0` before considering tail-coverage PASS. Keep `num_evals=6`, PPO batch size, source, checkpoint and push parameters unchanged. Check syntax/plan, check report config on next run, then check >=25/50/99 and gait. This is an experimental runner config change and requires fresh output/checkpoint SHA, NOT resume from old T1. Follow user rule EDIT-FIRST: **do not generate new runner/audit files**.

**M4.65-T1 continuous GPU verified (2026-11-21):** User supplied `m465_t1_continuous_20261121_075213.log` SHA256 `d0014528b502c292d4e029ef36bc4eb9170199ce3ee7e35cd42b3064cb2082bc`, `report(6).json` SHA256 `8dfeb8f3b776bdb9b009d82ebdeaad89885d82b133fbc8a70032199c168f3b3c`, and `tail_audit(1).json` SHA256 `ea0176867bf8a2eb1b2edd038b23cecc84060c3cfda7edff62859bfa0d07e130`. All reported technical and telemetry checks PASS, `hardware_scaled_config.num_resets_per_eval=0`, five PPO rollouts (1,310,720 env steps), 7,170 new pushes (per rollout 6500/68/300/204/98), 649,209 measured post-push action transitions, milestone counts >=25: 7,018; >=50: 6,580; >=99: 5,896; peak 8,945/12,288 MiB. These are action-exposure counters, **not proven balance recovery**; single push/episode in `easy` explains fewer new events after rollout 1. Two-env randomized source eval reward changes 20.784→6.188, avg episode length 726.5→337.5: possible gait warning; NOT proof of regression. Final P checkpoint name `m463_easy_probe5_step_001310720.msgpack`, SHA256 `c2c380d93cba1bfba2c0c88931d8b6854a40f5905ea6fd8db6d91b4019da08f1`; SHA was checked by user's local audit; bytes not independently uploaded. **Decision: continuous post-push exposure technical PASS / robustness PENDING; long PPO pilot NO-GO; get-up NOT IMPLEMENTED.** EDIT-FIRST next task: reuse `m463_gate3_cp_eval.py`; broaden its hardcoded smoke-report validator to accept pinned, SHA-verified `probe5` with 5 checkpoints and `num_resets_per_eval=0`; change output `train_steps_P` from hardcoded 262,144 to report steps; run small fixed-seed C vs final P (1,310,720) matched-ON/OFF evaluation and inspect nominal gait/survival/recovery before budget. No new scripts/files needed beyond user-generated evaluation artifacts. Offline patch semantics tested: accepts smoke & probe5; rejects bad SHA and host reset=1. **Do not use legacy C/P reviewer or grid aggregator unchanged because their P SHA pins point to old smoke checkpoint.**

**NEXT TASK ƯU TIÊN:** M4.66 / M4.65-EVAL — `C_VS_CONTINUOUS_PROBE5_FIXED_COMMAND_GATE`: edit the **existing** `m463_gate3_cp_eval.py` report-validation logic in place (retain old smoke compatibility) and run paired ON/OFF on 401M C vs final M4.65-T1 P (1,310,720 additional steps), begin with 12 diagnostic policy cases; then only expand to development multi-seed 72 cases if the diagnostic is sound. No new PPO training before this. Preserve all checkpoint SHAs, evaluate nominal gait, survival, matched-OFF recovery and finite physics. Avoid legacy reviewer SHA mismatch. GET_UP/RESUME still unimplemented.

## 9. TASK LEDGER (ngắn, append-only)

| Ngày | Task | Bằng chứng | Quyết định | Rules |
|---|---|---|---|---|
| 2026-11-21 | M4.65-T1 continuous GPU telemetry / M4.66 C-P eval planned | 7170 PPO pushes; 649209 post-push transitions; >=25/50/99=7018/6580/5896; no host reset, P SHA c2c380d9…; small eval warning | Continuous exposure PASS / robustness untested / fixed-command C-vs-P next / long pilot NO-GO | 1.1.0 |
| 2026-11-21 | M4.65 T1 five-rollout GPU review + EDIT-FIRST governance | 32505 actual pushes; >=25 postpush actions 0; Brax host `num_resets_per_eval=1` resets env each epoch; 5 rollouts trained but not physically continuous | Technical PASS / exposure-horizon NO-GO / correct existing runner then re-test / long pilot NO-GO | 1.1.0 |
| 2026-11-21 | M4.64 T0 user audit + M4.65 T1 offline preparation | 6501 PPO pushes, 75377 subsequent actions, >=25 none; 9/9 offline tests for 5-rollout runner/audit, GPU T1 untested | T0 PASS/exposure only; T1 bounded PROBE GO; long pilot NO-GO; get-up NOT IMPLEMENTED | 1.0.0 |
| 2026-11-21 | M4.64-T0 audit-only metric naming hotfix | singular/plural mismatch verified in shipped scripts; 5 prior offline tests + alias tests PASS, GPU report not supplied | AUDIT FIX READY / WAIT FOR REAUDIT / NO PPO RERUN | 1.0.0 |
| 2026-11-21 | M4.64 offline C/P forensics + tail instrumentation preparation | 36 pairs verified; 2 discordant, P nominal flag 6.97%; 5/5 offline tests; T0 GPU pending | DATA PASS / TAIL T0 PENDING / PILOT NO-GO / F0-F1 DESIGN ONLY | 1.0.0 |
| 2026-11-21 | M4.63 Gate3b dose 0.50 | 72 case; C/P nominal 18/18; push 26/36 vs 26/36; matched-OFF recovery mixed | DATA PASS / ROBUSTNESS UNPROVEN / LONG PILOT NO-GO | 1.0.0 |
| 2026-11-21 | Research governance | Khóa mục tiêu push → fall → get-up → resume; định nghĩa thước đo và cổng nghiệm thu | RULES ACTIVE; GET-UP NOT IMPLEMENTED | 1.0.0 |

## 10. CHANGELOG (append changes; do not rewrite history)

- **v1.1.0 — 2026-11-21:** Thêm quy tắc bắt buộc EDIT-FIRST: ưu tiên chỉ rõ file/hàm/diff/lý do/test/rollback để người dùng tự sửa; chỉ tạo file mới khi có nhu cầu độc lập thực sự. Sửa trạng thái bằng chứng M4.65-T1: Brax host reset sau mỗi epoch khi `num_resets_per_eval=1`; 5 rollout cập nhật PPO nhưng không liên tục trạng thái vật lý. Đề xuất thay đúng 2 file runner/audit hiện có; chưa chạy lại GPU.
- **v1.0.0 — 2026-11-21:** Tạo single source-of-truth cho nghiên cứu G1. Mục tiêu mới bắt buộc: push exposure và recovery **cùng** tự đứng dậy sau ngã rồi tiếp tục command. Khóa cấu hình/baseline, phân biệt termination với fall, bổ sung Gate F0–F4, workflow từng task, và tích hợp Gate3b dose0.50 thực tế.

---

**Nguyên tắc bảo trì:** File này được phép tiến hóa có kiểm soát. Khi thông số, mục tiêu hay quy tắc thay đổi: **bump semantic version, ghi lý do + bằng chứng + ngày vào CHANGELOG, cập nhật CURRENT STATE, và commit vào đúng nhánh**. Các task phải khai báo version đã đọc. Nếu kết quả mới chỉ thay đổi số liệu, cập nhật CURRENT STATE/TASK LEDGER, không cần đổi khái niệm hoặc mục tiêu. Nếu thiếu file hoặc SHA, dừng kết luận và yêu cầu đúng bằng chứng thiếu.
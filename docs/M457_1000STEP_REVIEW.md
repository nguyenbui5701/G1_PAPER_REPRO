# M4.57 — Sim-to-Sim 1.000 bước (20 giây)

**Phạm vi:** checkpoint PPO seed 1; reset seed 5186; `G1JoystickFlatTerrain`; lệnh cố định `stand`, `forward`, `yaw_left`; push OFF và domain randomization OFF; MuJoCo CPU tự chạy `mujoco.mj_step` (10 physics substeps mỗi policy step), policy inference bằng JAX CUDA.

**JSON gốc:** `m457_run_seed5186_20261120_094043_665761.json`
**SHA-256 JSON gốc:** `4b8567eff33e46908836744b5cc2351efa96e7a7430b1cd8b9112eaf40a3c28f`
**SHA-256 checkpoint:** `1ccf579ea35d4518713b00ea20d77db8463bc469b7de903b8f10f457b3dc90b8`
**SHA-256 script:** `8438aef9a9ece0c694ac204953f931971f07a43582f9554f9d8e6213ecfdf7f9`

## Kết luận chính

**6/6 episode hoàn thành đủ 1.000 bước, không termination.** Đây là thành công chức năng Sim-to-Sim trong ba điều kiện nominal đã thử; **không** phải bằng chứng numerical parity, robustness nhiều seed, độ tự nhiên gait, hoặc paper-exact reproduction.

## Kết quả

| Lệnh | MJX steps | CPU steps | MJX tracking | CPU tracking | Chênh lệch CPU |
|---|---:|---:|---:|---:|---:|
| Stand — XY RMSE (m/s) | 1000 | 1000 | 0.077821 | 0.080251 | +3.12% |
| Forward — XY RMSE (m/s) | 1000 | 1000 | 0.151854 | 0.146518 | -3.51% |
| Yaw-left — yaw RMSE (rad/s) | 1000 | 1000 | 0.258818 | 0.275498 | +6.44% |

| Lệnh | MJX vận tốc tiến (m/s) | CPU vận tốc tiến (m/s) | MJX yaw (rad/s) | CPU yaw (rad/s) |
|---|---:|---:|---:|---:|
| stand | 0.00751 | 0.00613 | 0.01269 | 0.01736 |
| forward | 0.21309 | 0.22245 | 0.01756 | 0.01159 |
| yaw_left | 0.01275 | 0.02584 | 0.30450 | 0.29771 |

## Sai khác quỹ đạo cực đại trong toàn bộ 1.000 bước

Chú ý: `root_xyz_max_abs` là max chênh lệch **theo một trục tọa độ** ở mỗi thời điểm, không phải khoảng cách Euclid; quỹ đạo khác nhau không tự động là thất bại điều khiển.

| Lệnh | Root XYZ (cm) | Joint qpos (rad) | Joint qvel (rad/s) | Action |
|---|---:|---:|---:|---:|
| stand | 4.06 | 0.2750 | 8.233 | 1.068 |
| forward | 19.72 | 0.2781 | 12.232 | 1.460 |
| yaw_left | 23.27 | 0.5159 | 13.764 | 1.743 |

## Giới hạn và quyết định

- Một seed, ba commands, nominal push OFF; chưa chứng minh được trên nhiều seed hoặc khi có nhiễu/push.
- MuJoCo CPU tự tích phân physics nhưng inference vẫn là JAX CUDA, không phải chạy toàn bộ trên CPU.
- Hai engine có sai số ngay bước đầu và action khác về sau. Action đầu tiên được ép dùng observation chung nên `first_action_max_abs_difference=0` không phải chứng minh observation parity.
- Không thay đổi checkpoint, PPO, reward, model hoặc thông số vật lý để cố làm quỹ đạo khớp.
- **Chốt M4.57 (functional transfer, single-seed nominal)**; chuyển sang M4.58 — tổng hợp hồ sơ chứng cứ, kết quả, hạn chế và đối chiếu bài báo.

## Khuyến nghị Git

Chỉ đưa script M4.57 đã nghiệm thu cùng report và summary nhỏ vào branch nghiên cứu. Không commit file JSON thô nhiều MB hay checkpoint. Xác minh `git status`, `git diff --check`, `git diff --cached --stat` trước khi push.

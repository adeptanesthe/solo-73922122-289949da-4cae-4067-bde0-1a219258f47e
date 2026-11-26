# 蒙版传播 API · Mask Propagation API

用 **FastAPI + NumPy** 实现的动画中间帧蒙版传播服务。美术师只画好首尾（或若干
确认帧），服务依据相邻帧的**双向整数光流**把二值蒙版追溯传播到未确认帧，并明确
标出哪些像素仍然要补画。支持补交/更正任意帧、只重算相邻确认帧之间的区间、不可变
版本与同版本多帧一致查询、失败原子保留旧状态。

---

## 1. 快速开始

### Docker Compose（推荐）

```bash
docker compose up --build
# 打开 http://localhost:8000/docs
```

健康检查：`curl http://localhost:8000/health` → `{"status":"ok",...}`。

> 本仓库已把 Swagger UI 的 JS/CSS 资源 vendored 到 `app/static/`，所以即使浏览器
> 访问不了 CDN，`/docs` 也能正常打开。

### 本地虚拟环境（不用 Docker）

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

自测（另开终端，仅用标准库，可对 Docker 端口直接跑）：

```bash
python3 scripts/smoke_test.py http://localhost:8000
pip install -r requirements-dev.txt
pytest
```

---

## 2. 在 `/docs` 里演示「断链」和「补交」

### 方式 A：一键演示接口（最省事）

在 `/docs` 的 **demo** 分组：

* `POST /api/demo/chain-break`：直接返回一个 3 帧 × 3×5 的断链场景结果；
* `POST /api/demo/supplement`：一次返回补交第 2 帧前（v1）/后（v2）的完整多帧结果。

也可以命令行：

```bash
curl -s -X POST http://localhost:8000/api/demo/chain-break | python -m json.tool
curl -s -X POST http://localhost:8000/api/demo/supplement  | python -m json.tool
```

### 方式 B：标准接口手动点一遍（推荐演示路径）

1. `POST /api/projects`，Request body 的 **Examples** 下拉选
   **「断链演示（3 帧 · 3×5）」** → **Execute**，复制响应里的 `project_id`；
2. `GET /api/projects/{project_id}/frames` → Execute，查看：
   * `frames[1].status` 状态矩阵；
   * `needs_paint=1` 的两个像素：`(0,4)` 冲突、`(1,1)` 双链皆断；
   * 顶层 `pending_pixels_total = 2`；
3. （可选）打开 `GET /api/projects/{project_id}/sheet.svg` 或
   `.../frames/1/svg`，红格=`broken`、琥珀格=`conflict`，都是待补像素；
4. `PUT /api/projects/{project_id}/frames/1/mask`：
   * `frame_index` 填 `1`；
   * body 示例选 **「断链场景：第 1 帧成品蒙版」** → Execute；
   * 响应给出 `previous_version=1, version=2`；因为第 1 帧与首尾直接相邻，
     `recomputed_frames=[]`；
5. 再调 `GET .../frames`（当前版，`pending=0`）和
   `GET .../frames?version=1`（旧版仍可查，`pending=2`），对比新旧两版。

**局部重算演示（5 帧）**：用 `POST /api/projects` 的 **supplement** 示例建项，
然后对 `frame_index=2` 调 `PUT .../mask`（示例「补交第 2 帧」）。响应中
`recomputed_frames=[1,3]`，且
`pending_pixels_per_frame` 从 `[0,2,2,2,0]` 变为 `[0,0,0,2,0]`
——只重算了第 2 帧与左确认帧 0、右确认帧 4 之间的第 1、3 帧。再补交第 3 帧
（示例「再补交第 3 帧」）后全部清零。

---

## 3. 传播规则与数据约定

所有帧尺寸统一为 `height × width`，约束 `2 ≤ frame_count ≤ 40`、
`1 ≤ height,width ≤ 128`。蒙版为 0/1 二值。

相邻帧 `i ↔ i+1` 的光流边界 `flows[i]` 含两个 `H×W×2` 整数数组，最后一维为
**`[dy, dx]`**：

* `forward[r,c]`：帧 `i` 的 `(r,c)` → 帧 `i+1` 的 `(r+dy, c+dx)`；
* `backward[r,c]`：帧 `i+1` 的 `(r,c)` → 帧 `i` 的 `(r+dy, c+dx)`。

未确认帧的每个像素分别向左、右最近的确认帧追溯，每跨过一个边界：

1. 按方向取去程光流得到落点；落点越界 → 该侧链断；
2. 在落点取反向光流往返一次，往返坐标不等于本步起点 → 该侧链断
   （标准 forward-backward 一致性检查，整数流故为精确相等）。

两侧结果合并：

| 左链 | 右链 | 结果状态 | 取值 | 是否待补 |
|---|---|---|---|---|
| 通 | 通且同值 | `agree` (3) | 该值 | 否 |
| 通 | 通但异值 | `conflict` (4) | -1 | **是** |
| 通 | 断 | `left` (1) | 左值 | 否 |
| 断 | 通 | `right` (2) | 右值 | 否 |
| 断 | 断 | `broken` (0) | -1 | **是** |
| 确认帧 | — | `confirmed` (5) | 提交的蒙版 | 否 |

`mask` 矩阵中待补像素用 `-1` 表示；`needs_paint` 是 0/1 待补矩阵；`status` 是
可读状态字符串矩阵，`status_codes` 是整数版。

引擎用 NumPy 从确认锚点向外增量合成「坐标映射」，每个区间每条边界只做一次
向量化处理，语义与逐像素独立追溯完全等价（测试里有暴力 oracle 随机对照）。

---

## 4. 版本、并发与失败原子性

* 每次建项 / 成功提交都生成一个新的**不可变 `Snapshot`**（完整
  `values` / `status` / `confirmed` 数组），版本号单调递增，历史版本全部保留，
  可用 `?version=N` 查询。
* 提交蒙版时：先校验，再在**副本**上只重算 `(L, k)` 与 `(k, R)` 两个相邻区间
  （`L/R` 为提交帧两侧最近确认帧，不存在则跳过该侧），成功后才在写锁内原子替换
  当前版本指针。
* 读接口不获取写锁，只一次性读取当前指针或指定的不可变版本，然后仅用这一个快照
  构造整个多帧响应——所以**一次多帧查询的所有帧必定来自同一版**，重算进行中也能
  立即读到完整旧版。路由均为同步 `def`，CPU/等待工作在线程池执行，不会阻塞事件循环。
* 输入校验（Pydantic + 形状/二值/整数流校验）全部在改动状态前完成；重算异常时
  丢弃副本、不移动版本指针，原确认帧与传播结果保持不变，接口返回 422（校验）或
  500（重算）。有专门的故障注入测试验证这一点。

---

## 5. HTTP 接口一览

| 方法 & 路径 | 说明 |
|---|---|
| `POST /api/projects` | 提交帧数/尺寸/首尾蒙版/双向光流，创建 v1 并返回全部帧 |
| `GET  /api/projects/{id}` | 项目元信息（当前版本、确认帧列表） |
| `GET  /api/projects/{id}/frames?version=` | 某一版的全部帧（同一快照，含统计） |
| `GET  /api/projects/{id}/frames/{f}?version=` | 单帧结果 |
| `PUT  /api/projects/{id}/frames/{f}/mask` | 补交/更正某确认帧蒙版，生成新版并只重算相邻区间 |
| `GET  /api/projects/{id}/frames/{f}/svg` | 单帧 SVG 色块图（红=断链，琥珀=冲突） |
| `GET  /api/projects/{id}/sheet.svg?version=` | 全部帧横向总览 SVG |
| `POST /api/demo/chain-break` | 一键断链演示 |
| `POST /api/demo/supplement` | 一键补交 + 局部重算 + 新旧版本对比 |
| `GET /health` · `GET /docs` · `GET /redoc` | 健康检查 / Swagger / ReDoc |

---

## 6. 目录结构

```
app/
  propagation.py   # 纯 NumPy 传播引擎 + 输入数组规范化/校验 + 状态常量
  store.py         # 项目仓库、不可变 Snapshot、写锁、事务式提交、版本指针
  schemas.py       # Pydantic 请求模型与边界约束
  serialization.py # 快照 -> JSON 响应
  render.py        # 零依赖 SVG 渲染
  demo.py          # 两个确定性演示场景（含精心设计的断链光流）
  main.py          # FastAPI 路由、异常映射、自托管 /docs
  static/          # vendored Swagger UI 资源 + favicon
tests/             # 113 个测试（引擎 oracle / store / HTTP / 真 socket 并发）
scripts/smoke_test.py  # 仅标准库的 27 项真实 HTTP 端到端断言
Dockerfile · docker-compose.yml · requirements*.txt
```

---

## 7. 自测与实测结果（交付前）

以下结果在本仓库随附的 Python 3.11 venv 中实际运行得到：

* **单元/集成测试：`113 passed`**（约 2.7 s）
  * `tests/test_propagation.py` **86 项**：固定场景（越界、往返不一致、单侧、
    一致非零流、冲突）、**80 组随机用例与独立暴力逐像素 oracle 全矩阵比对**、
    子区间不越界写、40×128×128 性能；
  * `tests/test_store.py`：重算区间精确性、最近确认帧更新、更正已确认帧、
    端点只有单侧区间、非法蒙版状态不变、注入重算故障后快照保留；
  * `tests/test_api.py`：`/docs`、OpenAPI 示例、精确状态矩阵、版本工作流、
    各类 422/404、故障 500 保状态、SVG；
  * `tests/test_concurrency.py`：起**真实 uvicorn socket**，用屏障让一次 PUT
    停在重算中，断言期间 GET 立即（<2 s）返回完整 v1 且每帧版本一致，放行后
    默认读切到 v2、`?version=1` 旧版仍在；另有 HTTP 层重算失败保状态。
* **真实 HTTP 冒烟：`ALL SMOKE CHECKS PASSED (27 assertions)`**
  （`python3 scripts/smoke_test.py http://127.0.0.1:8013`）。
* 实测演示输出（节选）：

  ```
  POST /api/demo/chain-break -> version 1 pending 2
      right    agree    agree    agree    conflict
      agree    broken   agree    left     agree
      agree    agree    agree    agree    left
  PUT frame/1/mask -> v1->v2 recomputed=[]
  current v2 pending=0 | old v1 pending=2
  supplement before [0, 2, 2, 2, 0] after [0, 0, 0, 2, 0] recomputed [1, 3]
  ```

* 性能/体积（本机）：最坏规模 40 帧 ×128×128，首次全区间传播约 **0.37 s**；
  中间插入一帧的两段重算约 **0.35 s**；`GET .../frames`（含字符串状态与 JSON
  编码）约 **0.9 s / 10.6 MB**；提交一个 128×128 蒙版（含 JSON 解析）约 **0.4 s**。
  小规模演示响应仅数 KB。

---

## 8. 剩余事项 / 已知边界（如实说明）

1. **本环境没有 Docker 二进制和 daemon，因此 `docker compose up --build` 未能在
   此处实际执行**；已做的核对是：`docker-compose.yml` 通过 YAML 解析、服务/构建
   上下文/健康检查配置完整，`Dockerfile` 使用标准 `python:3.11-slim` +
   `requirements.txt`，且同一 `uvicorn app.main:app` 已在本机真实起服务并通过
   全部 HTTP 冒烟。首次在有 Docker 的机器上仍建议跑一次
   `docker compose up --build` 与 `python3 scripts/smoke_test.py`。
2. 存储是**进程内内存态**，重启即丢，也没有多副本共享；历史版本目前无限保留。
   对当前单机/演示规模（单项目 ≤ 约 65.5 万像素）足够；生产化可加 SQLite/对象存储
   与版本淘汰策略。
3. 无鉴权/多租户，默认监听 `0.0.0.0:8000`，仅限可信网络或自行加网关。
4. 光流按需求假定为**整数位移**且所有帧同尺寸；不支持亚像素流、逐帧不同分辨率、
   蒙版删除确认帧。
5. 最大规格下带字符串状态的全量 JSON 约 10 MB；超大 `sheet.svg` 元素较多
   （已对大网格缩小格子并省略 tooltip），大规模巡检建议用单帧 SVG 或 JSON。
6. `/redoc` 仍走 CDN 资源（`/docs` 的 Swagger 已完全本地化）；离线环境若需要
   ReDoc，可同样 vendoring `redoc.standalone.js`。

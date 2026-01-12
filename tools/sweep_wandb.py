import wandb

api = wandb.Api()
runs = api.runs("xnf/MeshFlow2")  # 你的路径

print(f"Checking {len(runs)} runs...")

for run in runs:
    # --- 1. 安全获取属性 ---
    # getattr(对象, '属性名', 默认值)
    # 如果 run 没有 duration 属性，就默认为 0
    duration = getattr(run, 'duration', 0)
    
    # 同样防止 step 读取失败
    step = run.summary.get("_step", 0)

    # --- 2. 判断条件 ---
    # A: 状态是 crashed 或 failed
    is_bad_state = run.state in ["crashed", "failed"]
    
    # B: 运行时间极短 (比如小于 20秒)
    # 注意：有些 crashed 的 run duration 可能显示为 0
    is_short = duration < 20
    
    # C: 没有跑任何 step
    is_empty = step == 0

    # --- 3. 执行删除 ---
    if is_bad_state or (is_short and is_empty):
        print(f"Deleting: {run.name} | State: {run.state} | Time: {duration}s")
        run.delete()

print("Done.")
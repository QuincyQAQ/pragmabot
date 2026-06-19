"""
PragmaBot Fast: Merge Scene+Plan into one API call, saving one network round-trip.
"""
import sys, os, time, cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pragmabot", "src"))

from openai import OpenAI
from pragmabot.vlm_client import VLMClient
from pragmabot.vlm_task_planner import VLMTaskPlanner
from pragmabot.vlm_success_detector import VLMSuccessDetector
from pragmabot.vlm_exp_summarizer import VLMExperienceSummarizer
from pragmabot.memory_manager import MemoryManager

# ---- Config ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
CLIENT = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://yunwu.ai/v1",
)
MAX_STEPS = 10


class BigCfg:
    """Plan — use large model for reasoning."""
    vlm_model = "gpt-5.4"
    text_embedding_model = "text-embedding-3-large"

class FastCfg:
    """Detect/Summary — use lightweight model for speed."""
    vlm_model = "gpt-5.4-mini"
    text_embedding_model = "text-embedding-3-large"


def capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Camera 0 not available, trying camera 1...")
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Warning: No camera found. Returning a blank test image.")
        cap.release()
        return Image.new("RGB", (640, 480), color=(128, 128, 128))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("Warning: Camera read failed. Returning a blank test image.")
        return Image.new("RGB", (640, 480), color=(128, 128, 128))
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def main():
    print("PragmaBot Fast (merged Scene+Plan | Plan: gpt-5.4 | Detect: gpt-5.4-mini)")
    vlm_big = VLMClient(CLIENT, BigCfg())
    vlm_fast = VLMClient(CLIENT, FastCfg())
    conv_log = []

    planner = VLMTaskPlanner(vlm_big, conv_log)        # Plan 用大模型
    detector = VLMSuccessDetector(vlm_fast, conv_log)   # Detect 用 mini
    summarizer = VLMExperienceSummarizer(vlm_fast, conv_log)

    mem_cfg = type("Cfg", (), {"vlm_model": "gpt-5.4", "text_embedding_model": "text-embedding-3-large"})()
    memory = MemoryManager(VLMClient(CLIENT, mem_cfg), conv_log)

    print(f"Memory: {memory.n_ltm_entries} experiences")
    print("Enter=new task  q=quit")
    print("=" * 60)

    while True:
        cmd = input(f"\nTask > ").strip()
        if cmd.lower() == 'q':
            break
        task = cmd if cmd else "Observe the environment"

        img = capture()
        if img is None:
            print("Camera error")
            continue

        # Step 1+2+3 合并：Plan 本身就包含了场景理解
        # Plan 输出有 scene_description 字段，代替独立的 Scene 调用
        t0 = time.time()
        ltm = []
        if memory.n_ltm_entries > 0:
            # 用任务文本做一次快速场景 key（不需要 VLM 描述）
            # Memory 检索失败也继续
            pass
        plan, rt, _ = planner.plan_action(task, img, [], ltm)
        scene = plan.scene_description  # Plan 自带场景描述！
        print(f"\n[1+3 Plan  {rt:.1f}s]")
        print(f"  Scene: {scene[:200]}")
        print(f"  Action: {plan.chosen_action}")
        print(f"  Skill:  {plan.chosen_skill.value} | Target: {plan.target_object}")
        if plan.chain_of_thought_reasoning:
            print(f"  CoT:    {plan.chain_of_thought_reasoning[:200]}")

        # Step 2: Memory (在 Plan 后做，用 Plan 的 scene_description)
        if memory.n_ltm_entries > 0:
            try:
                ltm, _, _, _, _ = memory.retrieve_relevant_experiences(task, scene, top_k=3)
                if ltm:
                    print(f"[2. Memory] {len(ltm)} relevant (for next Plan)")
            except Exception:
                pass

        # Steps 4-6 loop
        stm = []
        task_done = False
        det_override = False
        for step in range(1, MAX_STEPS + 1):
            print(f"\n--- Step {step} ---")

            # Re-Plan with STM if not first step
            if step > 1:
                t0 = time.time()
                plan, rt, _ = planner.plan_action(task, img, stm, ltm)
                print(f"[3. Re-Plan {rt:.1f}s] {plan.chosen_action}")
                print(f"  Skill: {plan.chosen_skill.value} | Target: {plan.target_object}")

            # Execute
            print(f"[4. Execute] Enter=simulate fail | o=pretend success | q=quit")
            before = img
            user = input("  > ").strip().lower()
            if user == 'q':
                return
            if user == 'o':
                # 假装成功：对 Detect 用同一张图但标记为完成
                after = before
                det_override = True
            else:
                after = capture() or before
                det_override = False

            # Detect
            if det_override:
                print(f"[5. Detect] (skipped — manual OK)")
                action_ok, task_ok = True, True
            else:
                t0 = time.time()
                det = detector.perform_success_detection(task, plan.chosen_action, before, after)
                print(f"[5. Detect {time.time()-t0:.1f}s]")
                action_ok, task_ok = det.is_action_successful, det.is_task_completed
                print(f"  OK: {action_ok} | Done: {task_ok}")
                print(f"  {det.scene_description[:150]}")

            # STM
            stm.append(f"Step {step}: {plan.chosen_action} | OK={action_ok}")
            print(f"[6. STM] {len(stm)} entries")

            img = after
            if task_ok:
                task_done = True
                break
            if not det_override:
                print(f"  Re-planning...")

        # Summary
        if task_done:
            print(f"\n[7. Summary] ...")
            try:
                summary = summarizer.summarize_stm_to_ltm(task, scene, stm)
                try:
                    memory.save_experience(task, scene, summary)
                except Exception:
                    pass
                print(f"  {summary[:200]}")
            except Exception as e:
                print(f"  failed: {e}")

        print(f"\nDone. Memory: {memory.n_ltm_entries}")


if __name__ == "__main__":
    main()

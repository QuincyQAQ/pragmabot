"""
PragmaBot 7-step pipeline: Scene → Memory → Plan → Execute → Detect → STM → Summary.

Usage:
  python camera_runner_lite.py                              # interactive mode
  python camera_runner_lite.py --task "pick the ball"       # single task, still interactive
  python camera_runner_lite.py --task "pick the ball" --auto  # fully autonomous
"""
import argparse
import sys, os, time, cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pragmabot", "src"))

from openai import OpenAI
from pragmabot.vlm_client import VLMClient
from pragmabot.vlm_scene_describer import VLMSceneDescriber
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


class VLMConfig:
    """gpt-5.4-mini for all VLM tasks."""
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
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    w, h = img.size
    if max(w, h) > 384:
        scale = 384 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


# Actions that indicate no manipulation is needed — skip the execute/detect loop
_PASSIVE_ACTIONS = {"do nothing", "observe", "continue observing",
                    "inspect", "wait", "none", "keep observing",
                    "inspect the scene", "look"}


def main(auto=False, task_arg=None, max_steps=MAX_STEPS):
    print("PragmaBot 7-step pipeline | gpt-5.4-mini (yunwu.ai)")
    cfg = VLMConfig()
    vlm = VLMClient(CLIENT, cfg)
    conv_log = []

    scene_d = VLMSceneDescriber(vlm, conv_log)           # Step 1: Scene
    planner = VLMTaskPlanner(vlm, conv_log)               # Step 3: Plan
    detector = VLMSuccessDetector(vlm, conv_log)          # Step 5: Detect
    summarizer = VLMExperienceSummarizer(vlm, conv_log)   # Step 7: Summary

    mem_cfg = type("Cfg", (), {"vlm_model": "gpt-5.4-mini", "text_embedding_model": "text-embedding-3-large"})()
    memory = MemoryManager(VLMClient(CLIENT, mem_cfg), conv_log)

    print(f"Memory: {memory.n_ltm_entries} experiences")
    if auto:
        print(f"Mode: autonomous | Task: {task_arg}")
    else:
        print("Enter=new task  q=quit")
    print("=" * 60)

    while True:
        if task_arg is not None:
            cmd = task_arg
            task_arg = None
        else:
            cmd = input(f"\nTask > ").strip()
        if cmd.lower() == 'q':
            break
        task = cmd if cmd else "Observe the environment"
        task_start = time.time()

        img = capture()
        if img is None:
            print("Camera error")
            continue

        # ---- Step 1: Scene ----
        t0 = time.time()
        scene = scene_d.get_scene_description(task, img)
        print(f"\n[1. Scene  {time.time()-t0:.1f}s]")
        print(f"  {scene}")

        # ---- Step 2: Memory ----
        ltm = []
        if memory.n_ltm_entries > 0:
            try:
                ltm, _, _, _, _ = memory.retrieve_relevant_experiences(task, scene, top_k=3)
                if ltm:
                    print(f"[2. Memory] {len(ltm)} relevant experience(s)")
            except Exception:
                pass

        # ---- Step 3: Plan ----
        t0 = time.time()
        plan, rt, _ = planner.plan_action(task, img, [], ltm)
        print(f"[3. Plan  {rt:.1f}s]")
        print(f"  Action: {plan.chosen_action}")
        print(f"  Skill:  {plan.chosen_skill.value} | Target: {plan.target_object}")
        if plan.chain_of_thought_reasoning:
            print(f"  CoT:    {plan.chain_of_thought_reasoning}")

        # Fast path: passive action → skip execute/detect loop
        if plan.chosen_action.strip().lower() in _PASSIVE_ACTIONS:
            print(f"[Passive] No manipulation needed — scene observed.")
            print(f"\nDone. Total: {time.time() - task_start:.1f}s | Memory: {memory.n_ltm_entries}")
            continue

        # ---- Steps 4-6: Execute → Detect → STM loop ----
        stm = []
        task_done = False
        det_override = False
        for step in range(1, max_steps + 1):
            print(f"\n--- Step 3-6 loop, iteration {step} ---")

            # Re-Plan with STM if not first iteration
            if step > 1:
                t0 = time.time()
                plan, rt, _ = planner.plan_action(task, img, stm, ltm)
                print(f"[3. Re-Plan {rt:.1f}s] {plan.chosen_action}")
                print(f"  Skill: {plan.chosen_skill.value} | Target: {plan.target_object}")

            # Step 4: Execute
            before = img
            if auto:
                print(f"[4. Execute] (auto: simulate fail)")
                after = capture() or before
                det_override = False
            else:
                print(f"[4. Execute] Enter=simulate fail | o=pretend success | q=quit")
                user = input("  > ").strip().lower()
                if user == 'q':
                    return
                if user == 'o':
                    after = before
                    det_override = True
                else:
                    after = capture() or before
                    det_override = False

            # Step 5: Detect
            if det_override:
                print(f"[5. Detect] (skipped — manual OK)")
                action_ok, task_ok = True, True
            else:
                t0 = time.time()
                det = detector.perform_success_detection(task, plan.chosen_action, before, after)
                print(f"[5. Detect {time.time()-t0:.1f}s]")
                action_ok, task_ok = det.is_action_successful, det.is_task_completed
                print(f"  OK: {action_ok} | Done: {task_ok}")
                print(f"  {det.scene_description}")

            # Step 6: STM
            stm.append(f"Step {step}: {plan.chosen_action} | OK={action_ok}")
            print(f"[6. STM] {len(stm)} entries")

            img = after
            if task_ok:
                task_done = True
                break
            if not det_override:
                print(f"  Task not done, re-planning...")

        # ---- Step 7: Summary → LTM ----
        if task_done:
            print(f"\n[7. Summary] ...")
            try:
                summary = summarizer.summarize_stm_to_ltm(task, scene, stm)
                try:
                    memory.save_experience(task, scene, summary)
                except Exception:
                    pass
                print(f"  {summary}")
            except Exception as e:
                print(f"  failed: {e}")
        else:
            print(f"\n[7. Summary] skipped — task not completed in {max_steps} steps")

        print(f"\nDone. Total: {time.time() - task_start:.1f}s | Memory: {memory.n_ltm_entries}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PragmaBot 7-step pipeline — VLM robot task planner")
    parser.add_argument("--task", type=str, default=None, help="Task instruction (skips prompt)")
    parser.add_argument("--auto", action="store_true", help="Autonomous mode: skip all interactive prompts")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help=f"Max action steps (default: {MAX_STEPS})")
    args = parser.parse_args()
    main(auto=args.auto, task_arg=args.task, max_steps=args.max_steps)

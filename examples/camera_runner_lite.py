"""
PragmaBot pipeline (6-step data flow, maps to original 7-step logical model).

Data flow:
  1. Memory      — task text → embedding → LTM retrieval
  2. Scene+Plan  — one API call: image + task + LTM → scene + action
  3. Execute     — perform the action
  4. Detect      — compare before/after images
  5. STM         — record result; loop 2–5 if not done
  6. Summary     — distill STM → LTM

Logical flow (original 7-step model):
  Scene → Memory → Plan → Execute → Detect → STM → Summary
  (Scene+Plan are merged into one call since both need the same image)

Usage:
  python camera_runner_lite.py                              # interactive mode
  python camera_runner_lite.py --task "pick the ball"       # single task, still interactive
  python camera_runner_lite.py --task "pick the ball" --auto    # fully autonomous
  python camera_runner_lite.py --task "pick the ball" --one-shot # observe+plan only, no loop
"""
import argparse
import hashlib
import json
import sys, os, time, cv2
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pragmabot", "src"))

from openai import OpenAI
from pragmabot.vlm_client import VLMClient
from pragmabot.vlm_task_planner import VLMTaskPlanner
from pragmabot.vlm_success_detector import VLMSuccessDetector
from pragmabot.vlm_exp_summarizer import VLMExperienceSummarizer
from pragmabot.memory_manager import MemoryManager

# ---- Config ----
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")
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


# ---- VLM response cache (survives Ctrl+C, cleared on task completion) ----
def _img_hash(img: Image.Image) -> str:
    """Perceptual hash — same scene → same hash."""
    gray = img.convert("L").resize((32, 32), Image.LANCZOS)
    pixels = list(gray.getdata())  # noqa (compatible)
    avg = sum(pixels) // len(pixels)
    bits = "".join("1" if p > avg else "0" for p in pixels)
    return hashlib.md5(bits.encode()).hexdigest()[:16]


def _cache_key(task: str, img: Image.Image, step: str) -> str:
    """Unique key per task+image+step."""
    raw = f"{task}|{_img_hash(img)}|{step}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_get(task: str, img: Image.Image, step: str) -> dict | None:
    """Read cached VLM response, or None if missing/expired."""
    key = _cache_key(task, img, step)
    path = os.path.join(_CACHE_DIR, key + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _cache_set(task: str, img: Image.Image, step: str, data: dict) -> None:
    """Write VLM response to cache."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    key = _cache_key(task, img, step)
    path = os.path.join(_CACHE_DIR, key + ".json")
    with open(path, "w") as f:
        json.dump(data, f)


def _cache_clear() -> None:
    """Remove all cache files after task completion."""
    if os.path.isdir(_CACHE_DIR):
        for f in os.listdir(_CACHE_DIR):
            os.remove(os.path.join(_CACHE_DIR, f))
        print(f"[Cache] cleared")


def main(auto=False, task_arg=None, max_steps=MAX_STEPS, one_shot=False):
    print("PragmaBot 7-step pipeline | gpt-5.4-mini (yunwu.ai)")
    cfg = VLMConfig()
    vlm = VLMClient(CLIENT, cfg)
    conv_log = []

    planner = VLMTaskPlanner(vlm, conv_log)               # Scene+Plan merged
    detector = VLMSuccessDetector(vlm, conv_log)          # Detect
    summarizer = VLMExperienceSummarizer(vlm, conv_log)   # Summary

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

        # ---- Step 1: Memory (task text → LTM, no scene needed) ----
        ltm = []
        if memory.n_ltm_entries > 0:
            try:
                ltm, _, _, _, _ = memory.retrieve_relevant_experiences(task, task, top_k=3)
                if ltm:
                    print(f"[1. Memory] {len(ltm)} relevant experience(s)")
            except Exception:
                pass

        # ---- Step 2: Scene+Plan merged (one API call, image + LTM → scene + action) ----
        cached = _cache_get(task, img, "sceneplan")
        if cached:
            from pragmabot.vlm_task_planner import NextBestAction
            plan = NextBestAction(**cached)
            rt = 0
            scene = plan.scene_description
            print(f"\n[2. Scene+Plan  cached]")
        else:
            t0 = time.time()
            plan, rt, _ = planner.plan_action(task, img, [], ltm)
            scene = plan.scene_description  # Plan output includes scene description
            _cache_set(task, img, "sceneplan", plan.model_dump(mode="json"))
            print(f"\n[2. Scene+Plan  {rt:.1f}s]")
        print(f"  Scene:  {scene}")
        print(f"  Action: {plan.chosen_action}")
        print(f"  Skill:  {plan.chosen_skill.value} | Target: {plan.target_object}")
        if plan.chain_of_thought_reasoning:
            print(f"  CoT:    {plan.chain_of_thought_reasoning}")

        # One-shot mode: stop after Scene+Plan, no execute/detect/loop
        if one_shot:
            print(f"\n[One-shot] Scene+Plan complete — no execution.")
            print(f"\nDone. Total: {time.time() - task_start:.1f}s | Memory: {memory.n_ltm_entries}")
            continue

        # Fast path: passive action → skip execute/detect loop
        if plan.chosen_action.strip().lower() in _PASSIVE_ACTIONS:
            print(f"[Passive] No manipulation needed — scene observed.")
            _cache_clear()
            print(f"\nDone. Total: {time.time() - task_start:.1f}s | Memory: {memory.n_ltm_entries}")
            continue

        # ---- Steps 3-5: Execute → Detect → STM loop ----
        stm = []
        task_done = False
        det_override = False
        for step in range(1, max_steps + 1):
            print(f"\n--- Loop iteration {step} ---")

            # Re-Plan with STM if not first iteration
            if step > 1:
                t0 = time.time()
                plan, rt, _ = planner.plan_action(task, img, stm, ltm)
                print(f"[ Re-Plan {rt:.1f}s] {plan.chosen_action}")
                print(f"  Skill: {plan.chosen_skill.value} | Target: {plan.target_object}")

            # Step 3: Execute
            before = img
            if auto:
                print(f"[3. Execute] (auto: simulate fail)")
                after = capture() or before
                det_override = False
            else:
                print(f"[3. Execute] Enter=simulate fail | o=pretend success | q=quit")
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
                print(f"[4. Detect] (skipped — manual OK)")
                action_ok, task_ok = True, True
            else:
                t0 = time.time()
                det = detector.perform_success_detection(task, plan.chosen_action, before, after)
                print(f"[4. Detect {time.time()-t0:.1f}s]")
                action_ok, task_ok = det.is_action_successful, det.is_task_completed
                print(f"  OK: {action_ok} | Done: {task_ok}")
                print(f"  {det.scene_description}")

            # Step 6: STM
            stm.append(f"Step {step}: {plan.chosen_action} | OK={action_ok}")
            print(f"[5. STM] {len(stm)} entries")

            img = after
            if task_ok:
                task_done = True
                break
            if not det_override:
                print(f"  Task not done, re-planning...")

        # ---- Step 6: Summary → LTM ----
        if task_done:
            print(f"\n[6. Summary] ...")
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
            print(f"\n[6. Summary] skipped — task not completed in {max_steps} steps")

        _cache_clear()
        print(f"\nDone. Total: {time.time() - task_start:.1f}s | Memory: {memory.n_ltm_entries}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PragmaBot 7-step pipeline — VLM robot task planner")
    parser.add_argument("--task", type=str, default=None, help="Task instruction (skips prompt)")
    parser.add_argument("--auto", action="store_true", help="Autonomous mode: skip all interactive prompts")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help=f"Max action steps (default: {MAX_STEPS})")
    parser.add_argument("--one-shot", action="store_true", help="Scene+Plan only, no execute/detect/loop")
    parser.add_argument("--sim", action="store_true", help="Shorthand for --auto --max-steps 2")
    args = parser.parse_args()
    if args.sim:
        args.auto = True
        args.max_steps = min(args.max_steps, 2)
    main(auto=args.auto, task_arg=args.task, max_steps=args.max_steps, one_shot=args.one_shot)

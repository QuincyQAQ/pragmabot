"""
PragmaBot non-ROS runner: OpenCV camera + GPT-5.5-Pro
Implements PragmaBot README 7-step pipeline exactly:
  1. SceneDescribe → 2. MemoryRetrieve → 3. Plan → 4. Execute →
  5. Detect → 6. STM+loop(3-6) → 7. Summarize→LTM
"""
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
OPENAI_CLIENT = OpenAI(
    api_key="sk-bYkj2zaI3qd8fISMOGe3BOQLBfyOmowH9YUQUnNsi7zUNr5m",
    base_url="https://yunwu.ai/v1",
)
MAX_STEPS = 10

class VLMConfig:
    vlm_model = "gpt-5.4"
    text_embedding_model = "text-embedding-3-large"


def capture():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ret else None


def main():
    print("PragmaBot + GPT-5.4 (yunwu.ai)")
    cfg = VLMConfig()
    vlm = VLMClient(OPENAI_CLIENT, cfg)
    conv_log = []

    scene_d = VLMSceneDescriber(vlm, conv_log)
    planner = VLMTaskPlanner(vlm, conv_log)
    detector = VLMSuccessDetector(vlm, conv_log)
    summarizer = VLMExperienceSummarizer(vlm, conv_log)

    # Embeddings: same OpenAI client (if key supports it)
    emb_client = OPENAI_CLIENT
    emb_cfg = type("Cfg", (), {"vlm_model": "gpt-5.4", "text_embedding_model": "text-embedding-3-large"})()
    memory = MemoryManager(VLMClient(emb_client, emb_cfg), conv_log)

    print(f"Memory: {memory.n_ltm_entries} experiences")
    print("Enter=new task  q=quit")
    print("=" * 60)

    while True:
        cmd = input(f"\nTask > ").strip()
        if cmd.lower() == 'q':
            break
        task = cmd if cmd else "Observe the environment"

        # Step 1: Scene
        img = capture()
        if img is None:
            print("Camera error")
            continue
        t0 = time.time()
        scene = scene_d.get_scene_description(task, img)
        print(f"\n[1. Scene  {time.time()-t0:.1f}s]")
        print(f"  {scene}")

        # Step 2: Memory retrieval (skip on error)
        ltm = []
        if memory.n_ltm_entries > 0:
            try:
                ltm, _, _, _, _ = memory.retrieve_relevant_experiences(task, scene, top_k=3)
                if ltm:
                    print(f"[2. Memory] {len(ltm)} relevant experience(s)")
            except Exception as e:
                print(f"[2. Memory] skipped (embedding unavailable: {e})")
        else:
            print(f"[2. Memory] empty, skipping")

        # Steps 3-6 loop (STM self-reflection)
        stm = []
        task_done = False
        for step in range(1, MAX_STEPS + 1):
            print(f"\n--- Step 3-6 loop, iteration {step} ---")

            # Step 3: Plan
            t0 = time.time()
            plan, rt, _ = planner.plan_action(task, img, stm, ltm)
            print(f"[3. Plan  {rt:.1f}s]")
            print(f"  Action: {plan.chosen_action}")
            print(f"  Skill:  {plan.chosen_skill.value}")
            print(f"  Target: {plan.target_object}")
            if plan.chain_of_thought_reasoning:
                print(f"  CoT:    {plan.chain_of_thought_reasoning[:300]}")

            # Step 4: Execute
            print(f"\n[4. Execute] press Enter to simulate, or type action")
            before = img
            user_act = input("  > ").strip()
            if user_act.lower() == 'q':
                return
            after = capture() or before

            # Step 5: Detect success
            t0 = time.time()
            det = detector.perform_success_detection(task, plan.chosen_action, before, after)
            print(f"\n[5. Detect  {time.time()-t0:.1f}s]")
            print(f"  Action OK:    {det.is_action_successful}")
            print(f"  Task Done:    {det.is_task_completed}")
            print(f"  Scene change: {det.scene_description[:200]}")

            # Step 6: Update STM, loop or exit
            stm.append(f"Step {step}: {plan.chosen_action} | success={det.is_action_successful} | task_done={det.is_task_completed}")
            print(f"\n[6. STM] {len(stm)} entries accumulated")

            img = after  # use latest image for next Plan

            if det.is_task_completed:
                task_done = True
                break  # go to step 7
            else:
                print(f"  Task not done, STM self-reflection → re-planning...")

        # Step 7: Summarize to LTM
        if task_done:
            print(f"\n[7. Summary] distilling {len(stm)} STM entries to LTM...")
            try:
                summary = summarizer.summarize_stm_to_ltm(task, scene, stm)
                try:
                    memory.save_experience(task, scene, summary)
                except Exception:
                    pass  # embedding unavailable, skip LTM persistence
                print(f"  Saved: {summary[:200]}")
            except Exception as e:
                print(f"  Summary failed: {e}")
        else:
            print(f"\n[7. Summary] skipped — task not completed in {MAX_STEPS} steps")

        print(f"\nDone. Memory: {memory.n_ltm_entries} experiences")


if __name__ == "__main__":
    main()

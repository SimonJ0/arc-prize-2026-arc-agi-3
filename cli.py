"""
Interactive CLI for ARC-AGI-3 (ARC Prize 2026).
Enforces the Cryptographic Iron Rule submission workflow with strict command boundaries:
1. build-submission: Compile, bundle, and compute SHA-256 hashes only (Zero network).
2. request-approval: Produce formal authorization brief bound to hashes with 2h expiry token.
3. submit --approval <signed-token>: The ONLY command permitted to upload to Kaggle.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

from src.submit.submission_gate import SubmissionAuthorizationGate
from src.arc_core.metrics import EnvironmentEvaluation, LevelMetric
from src.arc_core.arc_loop import ArcResearchLoop
from agent.my_agent import MyAgent


def cmd_run_arc(args):
    """Executes the autonomous research loop over ARC-AGI-3 hypotheses."""
    loop = ArcResearchLoop(hypotheses_path=args.hypotheses)
    loop.run_all(force=args.force)



def cmd_build_submission(args):
    """Compiles agent bundle and builds Kaggle notebook without network calls."""
    gate = SubmissionAuthorizationGate()
    gate.build_submission(accelerator=args.accelerator)


def cmd_request_approval(args):
    """Produces signed authorization brief and approval token."""
    gate = SubmissionAuthorizationGate()
    gate.request_approval(
        exp_id=args.exp_id,
        eval_scorecard_path=args.eval_file,
        notes=args.notes or "",
    )


def cmd_submit(args):
    """Handles the 'Iron Rule' submission gate."""
    gate = SubmissionAuthorizationGate()

    if args.approval:
        print(f"\nHUMAN AUTHORIZATION TOKEN SUPPLIED: Initiating pre-upload verification...")
        gate.submit_with_approval(signed_token=args.approval)
    elif args.reject:
        exp_id = args.reject
        reason = args.reason or "No reason provided"
        print(f"Candidate {exp_id} REJECTED by human operator. Reason: {reason}")
    else:
        req_file = Path("reports/SUBMISSION_AUTHORIZATION_REQUEST.md")
        if req_file.exists():
            print(req_file.read_text(encoding="utf-8"))
        else:
            print("No pending submission authorization request found.")


def cmd_play(args):
    """Runs the agent locally on an official ARC-AGI-3 environment or synthetic micro-world."""
    from arcengine import GameAction, GameState

    if args.game != "synthetic_nav":
        try:
            import arc_agi
            arcade = arc_agi.Arcade()
            envs = arcade.get_environments()
            env_id = args.game
            matches = [e.game_id for e in envs if e.game_id.startswith(args.game)]
            if matches:
                env_id = matches[0]

            print(f"Loading official environment: {env_id} via Arcade...")
            env = arcade.make(env_id)
            agent = MyAgent(game_id=env_id)
            frame_data = env.reset()
            print(f"Initial State: {frame_data.state} | available_actions: {frame_data.available_actions}")

            for step in range(1, args.max_steps + 1):
                if agent.is_done(frame_data.frame, frame_data):
                    print(f"Agent finished at step {step} with state: {frame_data.state}")
                    break
                act = agent.choose_action(frame_data.frame, frame_data)
                print(f"  Step {step:02d}: Action -> {act.name}")
                frame_data = env.step(act)
                if frame_data.state == GameState.WIN:
                    print(f"\n[WIN] Level/Game completed at step {step}!")
                    break
            return
        except Exception as e:
            print(f"Notice: Unable to run on Arcade ({e}). Falling back to synthetic simulation...")

    print(f"Executing MyAgent locally on environment: {args.game}...")
    # Mock / synthetic run verification
    class SyntheticEnv:
        def __init__(self, game_id: str):
            self.game_id = game_id
            self.state = GameState.NOT_FINISHED
            self.levels_completed = 0
            self.available_actions = [0, 1, 2, 3, 4]
            self.guid = f"local-{game_id}"
            self.step_count = 0
            self.grid = np.zeros((20, 20), dtype=int)
            self.grid[10, 10] = 2  # Avatar
            self.grid[10, 14] = 3  # Goal

        @property
        def frame(self):
            return [self.grid]

        def step(self, action: GameAction):
            self.step_count += 1
            avatar_pos = np.argwhere(self.grid == 2)
            if len(avatar_pos) > 0:
                y, x = avatar_pos[0]
                if action == GameAction.ACTION4 and x < 19:
                    self.grid[y, x] = 0
                    self.grid[y, x + 1] = 2
                    if (y, x + 1) == (10, 14):
                        self.state = GameState.WIN

    env = SyntheticEnv(game_id=args.game)
    agent = MyAgent(game_id=args.game)

    print(f"Initial State: {env.state.value} | Start pos: (10, 10) | Target: (10, 14)")
    for step in range(1, args.max_steps + 1):
        if agent.is_done(env.frame, env):
            print(f"Agent finished at step {step} with state: {env.state.value}")
            break
        act = agent.choose_action(env.frame, env)
        print(f"  Step {step:02d}: Action -> {act.name}")
        env.step(act)
        if env.state == GameState.WIN:
            print(f"\n[WIN] Target reached in {step} actions!")
            break



def cmd_eval_platform(args):
    from src.arc_core.platform_bench import PlatformBenchmarkSuite
    suite = PlatformBenchmarkSuite()
    games_subset = [g.strip() for g in args.games.split(",")] if args.games else None
    suite.evaluate_suite(
        split=args.split,
        games_subset=games_subset,
        max_actions=args.max_actions,
        seed=args.seed,
    )


def main():
    parser = argparse.ArgumentParser(
        description="ARC-AGI-3 Autonomous Research & Submission CLI (ARC Prize 2026)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 0. Run Autonomous Research Loop
    arc_parser = subparsers.add_parser("run-arc", help="Launch autonomous self-improving research loop")
    arc_parser.add_argument("--hypotheses", default="configs/arc_hypotheses.yaml", help="Path to hypotheses yaml")
    arc_parser.add_argument("--force", action="store_true", help="Force re-evaluation of completed experiments")

    # 1. Build Submission
    build_parser = subparsers.add_parser("build-submission", help="Build and hash submission notebook (Zero network)")
    build_parser.add_argument("--accelerator", default="t4", choices=["cpu", "t4", "p100", "rtx6000"], help="Kaggle accelerator")

    # 2. Request Approval
    req_parser = subparsers.add_parser("request-approval", help="Generate signed authorization brief and token")
    req_parser.add_argument("--exp-id", required=True, help="Experiment ID to request authorization for")
    req_parser.add_argument("--eval-file", help="Path to evaluation scorecard file")
    req_parser.add_argument("--notes", help="Optional human-readable notes")

    # 3. Submit with Approval
    submit_parser = subparsers.add_parser("submit", help="Authorized Kaggle upload gate")
    submit_parser.add_argument("--approval", type=str, help="Signed HMAC authorization token")
    submit_parser.add_argument("--reject", type=str, help="Experiment ID to reject")
    submit_parser.add_argument("--reason", type=str, help="Reason for rejection")

    # 4. Play local
    play_parser = subparsers.add_parser("play", help="Play a local or synthetic game")
    play_parser.add_argument("--game", default="synthetic_nav", help="Game ID")
    play_parser.add_argument("--max-steps", type=int, default=30, help="Max physical steps")

    # 5. Evaluate on Official Platform
    plat_parser = subparsers.add_parser("eval-platform", help="Run official platform benchmark on real ARC-AGI-3 games")
    plat_parser.add_argument("--split", default="train", choices=["train", "holdout", "all"], help="Dataset split")
    plat_parser.add_argument("--games", type=str, help="Comma-separated game IDs (overrides split)")
    plat_parser.add_argument("--max-actions", type=int, default=100, help="Max actions per game")
    plat_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.command == "run-arc":
        cmd_run_arc(args)
    elif args.command == "build-submission":
        cmd_build_submission(args)
    elif args.command == "request-approval":
        cmd_request_approval(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "play":
        cmd_play(args)
    elif args.command == "eval-platform":
        cmd_eval_platform(args)


if __name__ == "__main__":
    main()

"""Valida a orquestração sem GPU, SLURM, Apptainer ou download de dados."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1]


class HammerComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hammer-pipeline-")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        scripts = self.repo / "scripts/ovx"
        scripts.mkdir(parents=True)
        for name in ("run_hammer_comparison.sh", "submit_hammer_comparison.sh", "load_wandb_env.sh"):
            shutil.copyfile(SOURCE / name, scripts / name)
        (self.repo / ".cache/mujoco_py_pkg").mkdir(parents=True)
        (self.repo / "neubay.sif").touch()
        (self.repo / "wandb.env").write_text("WANDB_API_KEY=test-only\n")
        self.pretrained = self.repo / "offline_world/ckpt/pretrained/adroit/hammer-cloned-v1"
        self.pretrained.mkdir(parents=True)
        for seed in range(3):
            (self.pretrained / f"ensemble_seed{seed}.eqx").write_text(json.dumps({
                "ensemble_size": 128, "hidden_size": 400, "has_ln": True,
            }) + "\npretrained-payload\n")
        (self.repo / "dataset.hdf5").write_bytes(b"fake-hdf5")
        fake = self.repo / "fake_tools"
        fake.mkdir()
        (fake / "gym.py").write_text('''
from pathlib import Path
class Space:
    shape = (26,)
class Env:
    observation_space = action_space = Space()
    dataset_filepath = str(Path("dataset.hdf5").resolve())
    def get_dataset(self): return {"rewards": [1, 2]}
    def close(self): pass
def make(name):
    assert name == "hammer-cloned-v1"
    return Env()
''')
        (fake / "d4rl.py").touch()
        apptainer = fake / "apptainer"
        apptainer.write_text('''#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys
args = sys.argv[sys.argv.index("bash") + 4:]
if args[0] == "-":
    result = subprocess.run([sys.executable, *args], input=sys.stdin.read(), text=True)
    sys.exit(result.returncode)
with open("calls.jsonl", "a") as log:
    log.write(json.dumps(args) + "\\n")
if os.environ.get("FAIL_WORLD") and args[0].endswith("cont_ensemble.py"):
    sys.exit(7)
options = dict(arg.lstrip("+").split("=", 1) for arg in args[1:] if "=" in arg)
seed = options["seed"]
if args[0].endswith("cont_ensemble.py"):
    output = pathlib.Path(options["ensemble.save_dir"]) / "hammer-cloned-v1" / f"ensemble_seed{seed}.eqx"
else:
    output = pathlib.Path(options["save_agent_root"]) / "adroit/hammer-cloned-v1" / f"agent_seed{seed}.eqx"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("new-checkpoint")
''')
        apptainer.chmod(0o755)
        sbatch = fake / "sbatch"
        sbatch.write_text('''#!/usr/bin/env python3
import json, sys
with open("submissions.jsonl", "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
print("6069")
''')
        sbatch.chmod(0o755)
        self.env = {
            **os.environ, "REPO_DIR": str(self.repo), "RUN_BATCH_ID": "test-hammer",
            "SLURM_ARRAY_TASK_ID": "1", "SLURM_JOB_ID": "6069",
            "PATH": str(fake) + os.pathsep + os.environ["PATH"],
            "PYTHONPATH": str(fake), "SMOKE_TEST": "false", "DRY_RUN": "false",
            "PRETRAINED_WM_DIR": "offline_world/ckpt/pretrained/adroit", "SEEDS": "0-2",
            "MAX_PARALLEL": "3", "TOTAL_EPOCHS": "1200",
        }
        self.env.pop("EXPECTED_GIT_COMMIT", None)
        for args in (["init", "-q"], ["add", "scripts"],
                     ["-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                      "-c", "commit.gpgsign=false", "commit", "-qm", "test fixture"]):
            subprocess.run(["git", *args], cwd=self.repo, check=True, capture_output=True)

    def run_script(self, script="run", **env):
        return subprocess.run(
            ["bash", f"scripts/ovx/{script}_hammer_comparison.sh"],
            cwd=self.repo, env={**self.env, **env}, text=True, capture_output=True,
        )

    def calls(self):
        path = self.repo / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_full_run_pairs_agents_and_preserves_pretrained(self):
        original = (self.pretrained / "ensemble_seed1.eqx").read_bytes()
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 3)
        self.assertTrue(calls[0][0].endswith("cont_ensemble.py"))
        self.assertIn("ensemble.total_epochs=1200", calls[0])
        agent_options = []
        for call in calls[1:]:
            self.assertEqual(call[0], "offline_cont.py")
            options = dict(arg.lstrip("+").split("=", 1) for arg in call[1:] if "=" in arg)
            self.assertEqual(options["seed"], "1")
            self.assertEqual(options["save_agent"], "true")
            # Apenas origem do WM e identificação dos artefatos podem diferir.
            for field in ("ensemble.save_dir", "ensemble.plot_dir", "save_agent_root", "exp_name", "wandb_tags"):
                options.pop(field)
            agent_options.append(options)
        self.assertEqual(agent_options[0], agent_options[1])
        self.assertFalse(any(arg.startswith("train.") for call in calls for arg in call))
        self.assertEqual(original, (self.pretrained / "ensemble_seed1.eqx").read_bytes())
        inputs = json.loads((self.repo / "logs/pipelines/test-hammer/seed1/inputs.json").read_text())
        self.assertEqual(inputs["sha256"], hashlib.sha256(b"fake-hdf5").hexdigest())
        self.assertEqual(len(list(self.repo.glob("offline_agent/ckpt/experiments/**/agent_seed1.eqx"))), 2)
        # Um reenvio não deve substituir resultados nem iniciar novos treinos.
        self.assertNotEqual(self.run_script().returncode, 0)
        self.assertEqual(len(self.calls()), 3)

    def test_smoke_changes_both_agents_and_isolates_outputs(self):
        result = self.run_script(SMOKE_TEST="true")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertIn("ensemble.total_size=2", calls[0])
        for call in calls[1:]:
            self.assertIn("collect.ensemble_size=2", call)
            self.assertIn("train.grad_steps=2000", call)
        self.assertEqual(len(list(self.repo.glob("offline_agent/ckpt/smoke/**/agent_seed1.eqx"))), 2)
        self.assertFalse((self.repo / "offline_agent/ckpt/experiments").exists())

    def test_failed_world_prevents_agent_training(self):
        result = self.run_script(FAIL_WORLD="true")
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertEqual(len(self.calls()), 1)
        self.assertFalse((self.repo / "offline_agent").exists())

    def test_missing_exact_seed_does_not_fall_back(self):
        (self.pretrained / "ensemble_seed1.eqx").unlink()
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_wrong_full_architecture_fails_before_training(self):
        (self.pretrained / "ensemble_seed1.eqx").write_text('{"hidden_size": 200}\n')
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_dry_run_does_not_submit(self):
        result = self.run_script("submit", DRY_RUN="true")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.repo / "submissions.jsonl").exists())
        self.assertFalse((self.repo / "logs").exists())

    def test_submission_records_array_and_rejects_duplicate_batch(self):
        result = self.run_script("submit")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        submissions = (self.repo / "submissions.jsonl").read_text()
        self.assertIn("--array=0-2%3", json.loads(submissions))
        plan = (self.repo / "logs/pipelines/test-hammer/jobs.tsv").read_text()
        self.assertIn("6069\t0-2\tfull", plan)
        self.assertNotEqual(self.run_script("submit").returncode, 0)
        self.assertEqual((self.repo / "submissions.jsonl").read_text(), submissions)


if __name__ == "__main__":
    unittest.main()

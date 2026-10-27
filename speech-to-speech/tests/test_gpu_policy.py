"""GPU inventory checks without allocating GPU memory."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_policy import validate_gpu, validate_stage_budgets

GIB = 1024**3


class SharedGpuTests(unittest.TestCase):
    def test_existing_stt_tts_can_already_occupy_memory(self):
        budget = validate_gpu(80 * GIB, 68 * GIB, (8, 0))
        self.assertEqual(budget["qwen_budget_gib"], 56)
        self.assertEqual(budget["other_services_budget_gib"], 24)
        self.assertEqual(budget["minimum_free_gib"], 60)

    def test_real_80gb_card_capacity_is_accepted(self):
        validate_gpu(79.15 * GIB, 65 * GIB, (9, 0))

    def test_smaller_gpu_cannot_be_passed_off_as_80gb(self):
        for total in (24, 40, 48):
            with self.subTest(total=total), self.assertRaisesRegex(RuntimeError, "80 ГБ"):
                validate_gpu(total * GIB, total * GIB, (8, 0))

    def test_incompatible_compute_capability_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "compute capability"):
            validate_gpu(80 * GIB, 80 * GIB, (7, 5))

    def test_insufficient_room_does_not_start_a_new_model(self):
        with self.assertRaisesRegex(RuntimeError, "60.0 GiB.*55.0 GiB"):
            validate_gpu(80 * GIB, 55 * GIB, (8, 0))

    def test_budget_drift_cannot_consume_the_stt_tts_reserve(self):
        stages = [SimpleNamespace(stage_id=i, devices="0", gpu_memory_utilization=fraction)
                  for i, fraction in enumerate((0.50, 0.15, 0.05))]
        validate_stage_budgets(stages)
        stages[0].gpu_memory_utilization = 0.60
        with self.assertRaisesRegex(RuntimeError, "Бюджеты"):
            validate_stage_budgets(stages)

    def test_all_stages_use_the_selected_shared_card(self):
        stages = [SimpleNamespace(stage_id=i, devices="0", gpu_memory_utilization=fraction)
                  for i, fraction in enumerate((0.50, 0.15, 0.05))]
        stages[1].devices = "1"
        with self.assertRaisesRegex(RuntimeError, "размещение"):
            validate_stage_budgets(stages)


if __name__ == "__main__":
    unittest.main()

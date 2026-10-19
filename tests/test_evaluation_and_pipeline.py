import json
from src.module_07_evaluation.benchmark_runner import run_lm_eval
from src.module_07_evaluation.results_visualizer import plot_benchmark_radar, plot_training_loss
from pipeline.generate_report import generate_markdown_report


def test_run_lm_eval_simulated_creates_results_json(tmp_path):
    out_dir = tmp_path / "eval_out"
    payload = run_lm_eval(
        model_id="Qwen/Qwen2.5-1.5B",
        tasks="mmlu,gsm8k",
        num_fewshot=0,
        batch_size=4,
        output_dir=str(out_dir),
        simulate=True,
    )
    res_file = out_dir / "results.json"
    assert res_file.exists()
    assert "mmlu" in payload["results"]
    assert "gsm8k" in payload["results"]


def test_plot_training_loss_and_radar_from_real_json(tmp_path):
    trainer_state = tmp_path / "trainer_state.json"
    trainer_state.write_text(
        json.dumps(
            {
                "log_history": [
                    {"epoch": 1.0, "loss": 2.1, "eval_loss": 2.0},
                    {"epoch": 2.0, "loss": 1.5, "eval_loss": 1.6},
                ]
            }
        ),
        encoding="utf-8",
    )
    loss_png = tmp_path / "loss.png"
    res = plot_training_loss(str(trainer_state), str(loss_png))
    assert res == str(loss_png)
    assert loss_png.exists()

    results_json = tmp_path / "results.json"
    results_json.write_text(
        json.dumps(
            {
                "results": {
                    "mmlu": {"acc,none": 0.55},
                    "gsm8k": {"exact_match,none": 0.42},
                }
            }
        ),
        encoding="utf-8",
    )
    radar_png = tmp_path / "radar.png"
    plot_benchmark_radar(str(results_json), str(radar_png))
    assert radar_png.exists()


def test_generate_markdown_report(tmp_path):
    report_file = tmp_path / "technical_report.md"
    out = generate_markdown_report(output_path=str(report_file))
    assert out == str(report_file)
    content = report_file.read_text(encoding="utf-8")
    assert "Vinicius Caridá (`vfcarida`)" in content
    assert "MMLU" in content
    assert "GSM8K" in content
